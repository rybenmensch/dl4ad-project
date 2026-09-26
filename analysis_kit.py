import csv
import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import numpy as np
import librosa

from lib import mean_absolute_error, mrstft
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
        return {"status": "CRASH", "error": "Reconstruction contains non-finite samples (NaN/Inf)"}
    
    # 2. Stille-Check: Wenn das Netz stirbt und nur noch Nullen ausgibt
    rms = torch.sqrt(torch.mean(audio_output**2))
    if rms < 1e-5:  # Entspricht ca. -100 dB (Totstille)
        return {"status": "SILENT", "error": f"Silence detected (RMS: {rms.item():.6f})"}
    
    # 3. Clipping-Erkennung: Nur als Flag mitgeben, NICHT löschen (Clipping ist beim Bending erwünscht!)
    clipping_count = (audio_output.abs() >= 0.99).sum().item()
    clipping_ratio = clipping_count / audio_output.numel()
    is_clipping = clipping_ratio > 0.05
    
    # 4. Standard-Metriken berechnen (MAE & STFT-Loss)
    mae = mean_absolute_error(baseline, audio_output)
    spectral = mrstft(audio_output, baseline)
    
    if not math.isfinite(mae) or not math.isfinite(spectral):
        return {"status": "CRASH", "error": "Comparison produced non-finite metrics"}
        
    features = extract_audio_features(audio_output, sr)
    
    return {
        "status": "VALID",
        "mae": mae,
        "mrstft": spectral,
        "clipping": is_clipping,
        **features,
        "error": ""
    }


def run_variant_evaluation(model, layer, variant, wav, baseline, sr, output_dir, stem, plots) -> dict:
    """Führt eine einzelne Modell-Modifikation aus und schickt das Audio durchs Quality Gate."""
    net = model.get_net(layer.net_type)
    row = {
        "layer_path": layer.layer_path,
        "layer_type": layer.name,
        "operation": variant.operation,
        "parameters": json.dumps(variant.parameters, sort_keys=True),
        "mae": None,
        "mrstft": None,
        "spectral_centroid": None,
        "spectral_bandwidth": None,
        "spectral_rolloff": None,
        "rms_energy": None,
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
        
        if eval_result["status"] != "VALID":
            raise ValueError(f"Quality Gate failed: {eval_result.get('error', eval_result['status'])}")
            
        row.update(
            mae=eval_result["mae"],
            mrstft=eval_result["mrstft"],
            clipping=eval_result["clipping"],
            spectral_centroid=eval_result["spectral_centroid"],
            spectral_bandwidth=eval_result["spectral_bandwidth"],
            spectral_rolloff=eval_result["spectral_rolloff"],
            rms_energy=eval_result["rms_energy"]
        )
        
        if output_dir is not None:
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
        row["error"] = f"{type(exc).__name__}: {exc}"
        
    return row


def analyze_module_impact(
    model: NNModel,
    wav: tuple[torch.Tensor, int],
    output_dir: str | Path,
    nets = ("encoder", "decoder"),
    plots: bool = True,
    slides: bool = True,
) -> list[dict]:
    if slides:
        plots = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.reset()
    
    try:
        model.model.eval()
        layers = []
        for net, net_type in model.get_nets_and_types():
            if net_type.value in nets:
                found = get_shape_preserving_layers_from_net(model, net)
                layers.extend(found)
                
        baseline = model(wav).detach().cpu().clone()
        sr = model.get_sample_rate()
        if output_dir is not None:
            save_audio(output_dir / "baseline.wav", baseline, sr)
            
        for layer in layers:
            # 1. Standard-Varianten durchtesten (Skip & Repeat in versch. Stärken)
            fixed_variants = [
                Variant("skip", {}),
                *[Variant("repeat", {"repeats": n}) for n in (3, 5, 10)]
            ]
            for variant in fixed_variants:
                model.reset()
                model.model.eval()
                stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_{variant.operation}"
                row = run_variant_evaluation(model, layer, variant, wav, baseline, sr, output_dir, stem, plots)
                rows.append(row)
                if output_dir is not None:
                    write_results(output_dir / "results.csv", rows)

            # 2. Multiplikationen mit adaptivem Sampling (Sucht nach Tipping Points)
            net = model.get_net(layer.net_type)
            if model.layer_has_weights(net[layer.index]):
                factors_to_test = [1.0, 5.0, 10.0, 20.0]
                tested_results = []
                
                for factor in factors_to_test:
                    model.reset()
                    model.model.eval()
                    variant = Variant("multiply", {"weight_mul": factor, "bias_mul": 1.0})
                    stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_multiply_{factor}"
                    row = run_variant_evaluation(model, layer, variant, wav, baseline, sr, output_dir, stem, plots)
                    rows.append(row)
                    tested_results.append((factor, row["mae"]))
                    if output_dir is not None:
                        write_results(output_dir / "results.csv", rows)
                
                # Wenn der Fehler zwischen zwei Testpunkten extrem springt, genauer hinschauen!
                for i in range(len(tested_results) - 1):
                    f1, mae1 = tested_results[i]
                    f2, mae2 = tested_results[i+1]
                    if mae1 is not None and mae2 is not None:
                        if abs(mae1 - mae2) > 0.5:  # Schwellenwert für Sprungerkennung
                            mid_factor = (f1 + f2) / 2.0
                            model.reset()
                            model.model.eval()
                            variant = Variant("multiply", {"weight_mul": mid_factor, "bias_mul": 1.0})
                            stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_multiply_adaptive_{mid_factor}"
                            row = run_variant_evaluation(model, layer, variant, wav, baseline, sr, output_dir, stem, plots)
                            rows.append(row)
                            if output_dir is not None:
                                write_results(output_dir / "results.csv", rows)

    finally:
        model.reset()
        
    rows.sort(key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True)
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