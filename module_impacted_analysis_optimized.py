import csv
import json
import math
import random
from pathlib import Path

import torch

from lib import mean_absolute_error, mrstft
from model import get_shape_preserving_layers_from_net, get_weighted_layers_from_net
from module_impacted_analysis import MODULES, save_audio


def adaptive_values(low, high, control, budget, rng, initial_samples=8):
    tested = []
    seen = set()
    initial = [control]
    count = min(initial_samples, budget - 1)
    initial.extend(low + (high - low) * (i + rng.random()) / count for i in range(count))
    for step in range(budget):
        value = None
        for attempt in range(100):
            if step < len(initial) and attempt == 0:
                candidate = initial[step]
            else:
                valid = sorted((score, x) for x, score in tested if score is not None and score > 0)
                if valid and rng.random() < 0.8:
                    anchors = [valid[round((len(valid) - 1) * q)][1] for q in (0.2, 0.5, 0.9)]
                    anchor = rng.choice(anchors)
                    positions = sorted({low, high, *seen})
                    index = positions.index(anchor)
                    left = positions[max(0, index - 1)]
                    right = positions[min(len(positions) - 1, index + 1)]
                    candidate = rng.uniform(left, right)
                else:
                    candidate = rng.uniform(low, high)
            candidate = round(candidate, 12)
            if candidate not in seen:
                value = candidate
                break
        if value is None:
            return
        seen.add(value)
        score = yield value
        tested.append((value, score))


def analyze_module_impact_optimized(
    model,
    wav,
    output_dir,
    max_artifacts=None,
    trials=24,
    seed=0,
    seconds=None,
    nets=("encoder", "decoder"),
    plots=True,
    slides=True,
    multiplier_range=(0.25, 8.0),
    addition_range=(-1.0, 1.0),
    max_gain_db=24.0,
):
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer of at least 2")
    if max_artifacts is not None and (isinstance(max_artifacts, bool) or not isinstance(max_artifacts, int) or max_artifacts < 0):
        raise ValueError("max_artifacts must be a nonnegative integer or None")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    if not 0 < multiplier_range[0] < 1 < multiplier_range[1] or not all(map(math.isfinite, multiplier_range)):
        raise ValueError("multiplier_range must be finite, positive, and contain 1")
    if not addition_range[0] < 0 < addition_range[1] or not all(map(math.isfinite, addition_range)):
        raise ValueError("addition_range must be finite and contain 0")
    if not math.isfinite(max_gain_db) or max_gain_db <= 0:
        raise ValueError("max_gain_db must be positive and finite")
    if not nets or set(nets) - {"encoder", "decoder"}:
        raise ValueError("nets must select encoder and/or decoder")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    if seconds is not None:
        wav = (wav[0][..., :int(seconds * wav[1])], wav[1])
    rows = []
    artifacts = []
    cache = {}
    plots = plots or slides

    def write_results():
        ranked = sorted(rows, key=lambda row: row["mrstft"] if row["mrstft"] is not None else -1, reverse=True)
        if ranked:
            with (output_dir / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(ranked[0]))
                writer.writeheader()
                writer.writerows(ranked)
        return ranked

    model.reset()
    try:
        model.model.eval()
        structural = []
        weighted = []
        for net, kind in model.get_nets_and_types():
            if kind.value in nets:
                structural.extend(get_shape_preserving_layers_from_net(model, net))
                weighted.extend(get_weighted_layers_from_net(model, net))
        if not structural and not weighted:
            raise ValueError("No eligible layers found")
        baseline = model(wav).detach().cpu().clone()
        if baseline.ndim != 2 or baseline.numel() == 0 or not torch.isfinite(baseline).all():
            raise ValueError("Baseline must be nonempty finite audio")
        baseline_rms = baseline.square().mean().sqrt().item()
        if baseline_rms < 1e-8:
            raise ValueError("Baseline is near-silent; select a different excerpt")
        sr = model.get_sample_rate()
        save_audio(output_dir / "baseline.wav", baseline, sr)
        settings = dict(trials_per_parameter=trials, seed=seed, seconds=seconds,
                        multiplier_range=multiplier_range, addition_range=addition_range,
                        max_gain_db=max_gain_db, ranking="mrstft", nets=nets)
        (output_dir / "search.json").write_text(json.dumps(settings, indent=2))

        def evaluate(layer, operation, parameters):
            key = (layer.layer_path, operation, json.dumps(parameters, sort_keys=True))
            if key in cache:
                return cache[key]
            row = dict(layer_path=layer.layer_path, layer_type=layer.name,
                       operation=operation, parameters=key[2], mae=None, mrstft=None,
                       gain_db=None, accepted=False, audio="", plot="", error="")
            stem = f"{len(rows):05d}_{layer.net_type.value}_{layer.index}_{operation}"
            score = None
            try:
                model.reset()
                model.model.eval()
                net = model.get_net(layer.net_type)
                net[layer.index] = MODULES[operation](layer, **parameters)
                net[layer.index].eval()
                reconstruction = model(wav).detach().cpu()
                if reconstruction.shape != baseline.shape:
                    raise ValueError("Reconstruction shape differs from baseline")
                if not torch.isfinite(reconstruction).all():
                    raise ValueError("Reconstruction contains non-finite values")
                rms = reconstruction.square().mean().sqrt().item()
                gain_db = 20 * math.log10(max(rms, 1e-12) / baseline_rms)
                row["gain_db"] = gain_db
                if rms < 1e-8 or gain_db < -60 or gain_db > max_gain_db:
                    raise ValueError("Output rejected by silence/gain limits")
                mae = mean_absolute_error(baseline, reconstruction)
                spectral = mrstft(reconstruction, baseline)
                if not math.isfinite(mae) or not math.isfinite(spectral):
                    raise ValueError("Non-finite comparison metrics")
                row.update(mae=mae, mrstft=spectral, accepted=True)
                score = spectral
                if max_artifacts != 0:
                    artifacts.append((row, stem, reconstruction.clone()))
                    artifacts.sort(key=lambda item: item[0]["mrstft"], reverse=True)
                    if max_artifacts is not None:
                        del artifacts[max_artifacts:]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            cache[key] = score
            write_results()
            return score

        for layer in structural:
            evaluate(layer, "skip", {})
            for repeats in (1, 2, 3, 5, 8):
                evaluate(layer, "repeat", {"repeats": repeats})
        for layer in weighted:
            model.reset()
            pairs = model.layer_get_weight_and_bias(layer.get_layer())
            scales = {}
            for target in ("weight", "bias"):
                tensors = [getattr(pair, target) for pair in pairs]
                tensors = [tensor for tensor in tensors if tensor is not None and tensor.numel()]
                if tensors:
                    total = sum(tensor.numel() for tensor in tensors)
                    scales[target] = math.sqrt(sum(tensor.detach().square().sum().item() for tensor in tensors) / total)
            for target, scale in scales.items():
                for operation in ("multiply", "add"):
                    if operation == "add" and scale == 0:
                        continue
                    low, high = (tuple(map(math.log2, multiplier_range)) if operation == "multiply" else addition_range)
                    candidates = adaptive_values(low, high, 0.0, trials, rng)
                    value = next(candidates)
                    while True:
                        if operation == "multiply":
                            parameters = {"weight_mul": 1.0, "bias_mul": 1.0}
                            parameters[f"{target}_mul"] = 2 ** value
                        else:
                            parameters = {"weight_add": 0.0, "bias_add": 0.0}
                            parameters[f"{target}_add"] = value * scale
                        score = evaluate(layer, operation, parameters)
                        try:
                            value = candidates.send(score)
                        except StopIteration:
                            break
    finally:
        model.reset()
    for row, stem, reconstruction in artifacts:
        try:
            path = output_dir / f"{stem}.wav"
            save_audio(path, reconstruction, sr)
            row["audio"] = path.name
            if plots:
                from plotting import plot_comparison

                path = output_dir / f"{stem}.png"
                plot_comparison(baseline, reconstruction, sr,
                                title=f"{row['layer_path']}: {row['operation']} {row['parameters']}",
                                save_path=str(path), show=False)
                row["plot"] = path.name
        except Exception as exc:
            row["error"] = f"Artifact export failed: {type(exc).__name__}: {exc}"
    ranked = write_results()
    if slides and artifacts:
        from quarto_slides import write_impact_slides

        write_impact_slides([item[0] for item in artifacts], output_dir, ranking="mrstft")
    return ranked
