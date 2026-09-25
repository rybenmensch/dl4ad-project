from cli.state import AnalyzeArgs, AppState
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from library.audio import mean_absolute_error, mrstft
from library.model import (
    NNModel,
    get_shape_preserving_layers_from_net,
    get_weighted_layers_from_net,
)
from library.layers import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer

MODULES = {
    "skip": SkippingLayer,
    "repeat": RepeatingLayer,
    "multiply": MultiplierLayer,
    "add": AdditionLayer,
}


@dataclass
class Variant:
    operation: str
    parameters: dict[str, int | float]

    def __post_init__(self):
        if self.operation not in MODULES:
            raise ValueError(f"Unknown operation: {self.operation}")


DEFAULT_ANALYSIS_VARIENTS = [
    Variant("skip", {}),
    # Total applications; 1 is the unchanged control.
    *[Variant("repeat", {"repeats": n}) for n in (3, 5, 10)],
    # Change weights while keeping biases unchanged.
    *[
        Variant("multiply", {"weight_mul": factor, "bias_mul": 1.0})
        for factor in (5.0, 10.0, 20.0)
    ],
    *[
        Variant("add", {"weight_add": 0.0, "bias_add": offset})
        for offset in (0.2, -0.2, 0.5, -0.5)
    ],
]


def analyze_module_impact(
    model: NNModel,
    wav: tuple[torch.Tensor, int],
    output_dir: str | Path,
    variants: list[Variant] = DEFAULT_ANALYSIS_VARIENTS,
    nets=("encoder", "decoder"),
    plots: bool = True,
    slides: bool = True,
    max_artifacts: int | None = None,
) -> list[dict]:
    if max_artifacts is not None and (
        isinstance(max_artifacts, bool)
        or not isinstance(max_artifacts, int)
        or max_artifacts < 0
    ):
        raise ValueError("max_artifacts must be a nonnegative integer or None")
    if not variants:
        raise ValueError("Provide at least one variant")
    if slides:
        plots = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    artifacts = []
    model.reset()
    try:
        model.model.eval()
        layers = {}
        eligible = {"skip": set(), "repeat": set(), "multiply": set(), "add": set()}
        operations = {variant.operation for variant in variants}
        for net, net_type in model.get_nets_and_types():
            if net_type.value not in nets:
                continue
            selections = []
            if operations & {"skip", "repeat"}:
                selections.append(
                    (
                        get_shape_preserving_layers_from_net(model, net),
                        ("skip", "repeat"),
                    )
                )
            if operations & {"multiply", "add"}:
                selections.append(
                    (get_weighted_layers_from_net(model, net), ("multiply", "add"))
                )
            for found, supported in selections:
                for layer in found:
                    layers[layer.layer_path] = layer
                    for operation in supported:
                        eligible[operation].add(layer.layer_path)
        if not layers:
            raise ValueError("No eligible layers found in the selected nets")
        baseline = model(wav).detach().cpu().clone()
        sr = model.get_sample_rate()
        if output_dir is not None:
            save_audio(output_dir / "baseline.wav", baseline, sr)
        for layer in layers.values():
            for variant in variants:
                if layer.layer_path not in eligible[variant.operation]:
                    continue
                model.reset()
                model.model.eval()
                net = model.get_net(layer.net_type)
                stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_{variant.operation}"
                row = {
                    "layer_path": layer.layer_path,
                    "layer_type": layer.name,
                    "operation": variant.operation,
                    "parameters": json.dumps(variant.parameters, sort_keys=True),
                    "mae": None,
                    "mrstft": None,
                    "audio": "",
                    "plot": "",
                    "error": "",
                }
                try:
                    net[layer.index] = MODULES[variant.operation](
                        layer, **variant.parameters
                    )
                    net[layer.index].eval()
                    reconstruction = model(wav).detach().cpu()
                    if reconstruction.shape != baseline.shape:
                        raise ValueError(
                            f"Output shape {tuple(reconstruction.shape)} differs from baseline {tuple(baseline.shape)}"
                        )
                    if not torch.isfinite(reconstruction).all():
                        raise ValueError("Reconstruction contains non-finite samples")
                    mae = mean_absolute_error(baseline, reconstruction)
                    # auraloss expects prediction first and reference second.
                    spectral = mrstft(reconstruction, baseline)
                    if not math.isfinite(mae) or not math.isfinite(spectral):
                        raise ValueError("Comparison produced non-finite metrics")
                    row.update(mae=mae, mrstft=spectral)
                    if max_artifacts != 0:
                        artifacts.append((row, stem, reconstruction.clone()))
                        artifacts.sort(key=lambda item: item[0]["mae"], reverse=True)
                        if max_artifacts is not None:
                            del artifacts[max_artifacts:]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                # Persist after each trial so long sweeps retain partial results.
                if output_dir is not None:
                    write_results(output_dir / "results.csv", rows)
    finally:
        model.reset()
    rows.sort(
        key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True
    )
    for row, stem, reconstruction in artifacts:
        try:
            audio_path = output_dir / f"{stem}.wav"
            save_audio(audio_path, reconstruction, sr)
            row["audio"] = audio_path.name
            if plots:
                from library.plotting import plot_comparison

                plot_path = output_dir / f"{stem}.png"
                plot_comparison(
                    baseline,
                    reconstruction,
                    sr,
                    title=f"{row['layer_path']}: {row['operation']} {row['parameters']}",
                    save_path=str(plot_path),
                    show=False,
                )
                row["plot"] = plot_path.name
        except Exception as exc:
            row["error"] = f"Artifact export failed: {type(exc).__name__}: {exc}"
    write_results(output_dir / "results.csv", rows)
    if slides and artifacts:
        from library.slides import write_impact_slides

        write_impact_slides([item[0] for item in artifacts], output_dir)
    return rows


def save_audio(path: Path, audio: torch.Tensor, sr: int) -> None:
    torchaudio.save(str(path), audio, sr, encoding="PCM_F", bits_per_sample=32)


def write_results(path: Path, rows: list[dict]) -> None:
    ranked = sorted(
        rows, key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ranked)


def analyze_loop(app: AppState) -> None:
    assert isinstance(app.args, AnalyzeArgs)

    for file in app.files:
        analyze_module_impact(
            model=app.model,
            wav=file.wav,
            output_dir=app.output,
            max_artifacts=app.args.save_depth,
        )
