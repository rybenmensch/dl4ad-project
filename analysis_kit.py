import csv
import json
import math
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import numpy as np
import librosa

from lib import mean_absolute_error, mrstft
from metrics import compute_error_metrics
from model import NNModel, get_shape_preserving_layers_from_net
from modules import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer

# Registrierte Module für das Network Bending
MODULES = {
    "skip": SkippingLayer,
    "repeat": RepeatingLayer,
    "multiply": MultiplierLayer,
    "add": AdditionLayer,
}


@dataclass
class Variant:
    """Describe one layer intervention and the arguments passed to its wrapper."""

    operation: str
    parameters: dict[str, int | float]

    def __post_init__(self):
        """Reject operations that have no registered intervention wrapper."""
        if self.operation not in MODULES:
            raise ValueError(f"Unknown operation: {self.operation}")


def extract_audio_features(audio: torch.Tensor, sr: int) -> dict:
    """Return mean spectral descriptors and RMS energy for a reconstructed waveform.

    Stereo input is averaged to mono for feature extraction. The returned values
    describe the output signal; they are not perceptual quality ratings.
    """
    # Auf Mono runterrechnen für librosa, falls Stereo reinkommt
    audio_np = audio.mean(dim=0).detach().cpu().numpy()
    
    return {
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=audio_np, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio_np, sr=sr))),
        "rms_energy": float(np.mean(librosa.feature.rms(y=audio_np)))
    }


def evaluate_transformation(audio_output: torch.Tensor, baseline: torch.Tensor, sr: int) -> dict:
    """Compare one reconstruction with its baseline and classify output health.

    Non-finite output and RMS below ``1e-5`` are returned as ``INVALID`` and
    ``SILENT`` respectively. Clipping is retained and flagged when more than
    5% of samples reach an absolute amplitude of 0.99. Successful comparisons
    include raw MAE, RMS/peak-normalized MAE, MR-STFT loss, and mean audio
    descriptors. The status describes technical signal conditions, not
    aesthetic quality.
    """
    # 1. Technischer Crash-Check: Verhindert, dass NaN/Inf die Statistiken sprengen
    if not torch.isfinite(audio_output).all():
        return {
            "status": "INVALID",
            "severity": 5,
            "error": "Reconstruction contains non-finite samples (NaN/Inf)",
        }
    
    # 2. Stille-Check: Wenn das Netz stirbt und nur noch Nullen ausgibt
    rms = torch.sqrt(torch.mean(audio_output**2))
    if rms < 1e-5:  # Entspricht ca. -100 dB 
        return {
            "status": "SILENT",
            "severity": 4,
            "error": f"Silence detected (RMS: {rms.item():.6f})",
        }
    
    # 3. Clipping-Erkennung: Nur als Flag mitgeben, NICHT löschen (Clipping ist beim Bending erwünscht!)
    clipping_count = (audio_output.abs() >= 0.99).sum().item()
    clipping_ratio = clipping_count / audio_output.numel()
    is_clipping = clipping_ratio > 0.05
    
    # 4. Standard-Metriken berechnen (MAE & STFT-Loss)
    mae = mean_absolute_error(baseline, audio_output)
    error_metrics = compute_error_metrics(
        baseline.detach().cpu().numpy(), audio_output.detach().cpu().numpy()
    )
    spectral = mrstft(audio_output, baseline)
    
    if not math.isfinite(mae) or not math.isfinite(spectral):
        return {
            "status": "INVALID",
            "severity": 5,
            "error": "Comparison produced non-finite metrics",
        }
        
    features = extract_audio_features(audio_output, sr)
    
    return {
        "status": "CLIPPED" if is_clipping else "VALID",
        "severity": 3 if is_clipping else 0,
        "mae": mae,
        "mae_normalized": error_metrics["mae_normalized"],
        "mrstft": spectral,
        "clipping": is_clipping,
        **features,
        "error": ""
    }


def run_variant_evaluation(
    model,
    layer,
    variant,
    wav,
    baseline,
    sr,
    output_dir,
    stem,
    plots,
    save_trial_audio=True,
) -> dict:
    """Apply one intervention, run inference, and return a result row.

    The selected layer is replaced in the model before inference. Shape changes,
    quality-gate outcomes, and runtime exceptions are recorded in the returned
    row instead of aborting the remaining sweep. Audio is saved only when
    ``save_trial_audio`` is true; plots are controlled separately by ``plots``.
    """
    net = model.get_net(layer.net_type)
    row = {
        "layer": layer.layer_path,
        "layer_path": layer.layer_path,
        "layer_type": layer.name,
        "operation": variant.operation,
        "parameters": json.dumps(variant.parameters, sort_keys=True),
        "mae": None,
        "mae_normalized": None,
        "mrstft": None,
        "spectral_centroid": None,
        "spectral_bandwidth": None,
        "spectral_rolloff": None,
        "rms_energy": None,
        "status": "ERROR",
        "severity": None,
        "clipping": False,
        "audio": "",
        "plot": "",
        "error": "",
    }
    
    try:
        net[layer.index] = MODULES[variant.operation](layer, **variant.parameters)
        net[layer.index].eval()
        reconstruction = model(wav).detach().cpu()
        
        if reconstruction.shape != baseline.shape:
            raise ValueError(f"Output shape differs from baseline")
        
        # Hier greift unser Quality Gate
        eval_result = evaluate_transformation(reconstruction, baseline, sr)

        row.update(
            status=eval_result["status"],
            severity=eval_result["severity"],
            mae=eval_result.get("mae"),
            mae_normalized=eval_result.get("mae_normalized"),
            mrstft=eval_result.get("mrstft"),
            clipping=eval_result.get("clipping", False),
            spectral_centroid=eval_result.get("spectral_centroid"),
            spectral_bandwidth=eval_result.get("spectral_bandwidth"),
            spectral_rolloff=eval_result.get("spectral_rolloff"),
            rms_energy=eval_result.get("rms_energy"),
        )

        if eval_result.get("error"):
            row["error"] = eval_result["error"]
        if eval_result["status"] in ("INVALID", "SILENT"):
            return row
        
        if output_dir is not None and save_trial_audio:
            audio_path = output_dir / f"{stem}.wav"
            save_audio(audio_path, reconstruction, sr)
            row["audio"] = audio_path.name
            
        if plots:
            from plotting import plot_comparison
            plot_path = output_dir / f"{stem}.png"
            plot_comparison(baseline, reconstruction, sr,
                            title=f"{layer.layer_path}: {variant.operation} {variant.parameters}",
                            save_path=str(plot_path), show=False)
            row["plot"] = plot_path.name
            
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
        
    return row


def analyze_module_impact(
    model: NNModel,
    wav: tuple[torch.Tensor, int],
    output_dir: str | Path,
    nets = ("encoder", "decoder"),
    plots: bool = True,
    slides: bool = True,
    interventions: tuple[str, ...] = ("skip", "repeat", "multiply"),
    add_offset: float = 0.1,
    save_baseline: bool = True,
    save_trial_audio: bool = True,
    write_intermediate_results: bool = True,
    write_results_csv: bool = True,
) -> list[dict]:
    """Sweep selected interventions over eligible layers for one audio input.

    A baseline reconstruction is computed once. Each trial starts from an
    in-memory copy of the pristine model when possible, falling back to the
    adapter's ``reset`` method if the model cannot be copied. Shape-preserving
    layers are considered for skip/repeat; weight-bearing layers are considered
    for multiply/add. Results are returned as dictionaries and can optionally
    produce baseline/trial audio, plots, slides, and incremental/final CSVs.

    Args:
        model: Model adapter that exposes encoder/decoder layers and inference.
        wav: Input waveform and its original sample rate.
        output_dir: Directory for requested result files and visualizations.
        nets: Network sections to inspect, normally ``encoder`` and ``decoder``.
        plots: Whether to save comparison plots for trials with exported audio.
        slides: Whether to write a slide deck from the ranked results.
        interventions: Enabled operation names from ``MODULES``.
        add_offset: Shared weight and bias offset used by the add intervention.
        save_baseline: Whether to export the unmodified reconstruction.
        save_trial_audio: Whether to export each successful trial reconstruction.
        write_intermediate_results: Rewrite the per-input CSV after every trial.
        write_results_csv: Write the final per-input CSV after the sweep.

    Returns:
        Trial rows, sorted by descending raw MAE; failed trials remain in the
        list with their status and error message.
    """
    if slides:
        plots = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown_interventions = set(interventions) - set(MODULES)
    if unknown_interventions:
        raise ValueError(f"Unknown interventions: {sorted(unknown_interventions)}")
    rows = []
    model.reset()
    pristine_model = None

    def restore_model() -> None:
        if pristine_model is None:
            model.reset()
        else:
            model.model = deepcopy(pristine_model)
            model.model.eval()
    
    try:
        model.model.eval()
        layers = []
        for net, net_type in model.get_nets_and_types():
            if net_type.value in nets:
                found = get_shape_preserving_layers_from_net(model, net)
                layers.extend(found)
                
        baseline = model(wav).detach().cpu().clone()
        sr = model.get_sample_rate()
        try:
            pristine_model = deepcopy(model.model)
        except Exception:
            pristine_model = None
        if output_dir is not None and save_baseline:
            save_audio(output_dir / "baseline.wav", baseline, sr)
            
        for layer in layers:
            fixed_variants = []
            if "skip" in interventions:
                fixed_variants.append(Variant("skip", {}))
            if "repeat" in interventions:
                fixed_variants.extend(
                    Variant("repeat", {"repeats": n}) for n in (3, 5, 10)
                )
            for variant in fixed_variants:
                restore_model()
                stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_{variant.operation}"
                row = run_variant_evaluation(
                    model, layer, variant, wav, baseline, sr, output_dir, stem,
                    plots, save_trial_audio
                )
                rows.append(row)
                if output_dir is not None and write_intermediate_results:
                    write_results(output_dir / "results.csv", rows)

            # 2. Multiplikationen mit adaptivem Sampling (Sucht nach Tipping Points)
            restore_model()
            net = model.get_net(layer.net_type)
            has_weights = model.layer_has_weights(net[layer.index])
            if has_weights and "multiply" in interventions:
                factors_to_test = [1.0, 5.0, 10.0, 20.0]
                tested_results = []
                
                for factor in factors_to_test:
                    restore_model()
                    variant = Variant("multiply", {"weight_mul": factor, "bias_mul": 1.0})
                    stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_multiply_{factor}"
                    row = run_variant_evaluation(
                        model, layer, variant, wav, baseline, sr, output_dir, stem,
                        plots, save_trial_audio
                    )
                    rows.append(row)
                    tested_results.append((factor, row["mae"]))
                    if output_dir is not None and write_intermediate_results:
                        write_results(output_dir / "results.csv", rows)
                
                # Wenn der Fehler zwischen zwei Testpunkten extrem springt, genauer hinschauen!
                for i in range(len(tested_results) - 1):
                    f1, mae1 = tested_results[i]
                    f2, mae2 = tested_results[i+1]
                    if mae1 is not None and mae2 is not None:
                        if abs(mae1 - mae2) > 0.5:  # Schwellenwert für Sprungerkennung
                            mid_factor = (f1 + f2) / 2.0
                            restore_model()
                            variant = Variant("multiply", {"weight_mul": mid_factor, "bias_mul": 1.0})
                            stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_multiply_adaptive_{mid_factor}"
                            row = run_variant_evaluation(
                                model, layer, variant, wav, baseline, sr, output_dir,
                                stem, plots, save_trial_audio
                            )
                            rows.append(row)
                            if output_dir is not None and write_intermediate_results:
                                write_results(output_dir / "results.csv", rows)

            if has_weights and "add" in interventions:
                restore_model()
                variant = Variant(
                    "add",
                    {"weight_add": add_offset, "bias_add": add_offset},
                )
                stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_add_{add_offset}"
                row = run_variant_evaluation(
                    model, layer, variant, wav, baseline, sr, output_dir, stem,
                    plots, save_trial_audio
                )
                rows.append(row)
                if output_dir is not None and write_intermediate_results:
                    write_results(output_dir / "results.csv", rows)

    finally:
        if pristine_model is None:
            model.reset()
        else:
            restore_model()
        
    rows.sort(key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True)
    if output_dir is not None and write_results_csv:
        write_results(output_dir / "results.csv", rows)
    if slides:
        from quarto_slides import write_impact_slides
        write_impact_slides(rows, output_dir)
    return rows


def save_audio(path: Path, audio: torch.Tensor, sr: int) -> None:
    """Write a waveform tensor as 32-bit floating-point PCM audio."""
    torchaudio.save(str(path), audio, sr, encoding="PCM_F", bits_per_sample=32)


def write_results(path: Path, rows: list[dict]) -> None:
    """Write trial rows to CSV, sorted by descending raw MAE.

    Rows without a computed MAE (for example, failed trials) are placed after
    measured rows.
    """
    if not rows:
        return
    ranked = sorted(rows, key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)