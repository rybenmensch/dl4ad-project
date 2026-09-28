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
    operation: str
    parameters: dict[str, int | float]

    def __post_init__(self):
        if self.operation not in MODULES:
            raise ValueError(f"Unknown operation: {self.operation}")


def extract_audio_features(audio: torch.Tensor, sr: int) -> dict:
    """Extrahiert objektive psychoakustische Audio-Deskriptoren mittels librosa."""
    # Auf Mono runterrechnen für librosa, falls Stereo reinkommt
    audio_np = audio.mean(dim=0).detach().cpu().numpy()
    
    return {
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=audio_np, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio_np, sr=sr))),
        "rms_energy": float(np.mean(librosa.feature.rms(y=audio_np)))
    }


def evaluate_transformation(audio_output: torch.Tensor, baseline: torch.Tensor, sr: int) -> dict:
    """
    Quality Gate & Evaluierungs-Funktion:
    Fängt technische Totalausfälle ab, lässt aber gewollten Glitch-Schmutz zu.
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
    """Führt eine einzelne Modell-Modifikation aus und schickt das Audio durchs Quality Gate."""
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
            if model.layer_has_weights(net[layer.index]) and "multiply" in interventions:
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

            if model.layer_has_weights(net[layer.index]) and "add" in interventions:
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
    torchaudio.save(str(path), audio, sr, encoding="PCM_F", bits_per_sample=32)


def write_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ranked = sorted(rows, key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)