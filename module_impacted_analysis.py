import csv
import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from lib import mean_absolute_error, mrstft
from model import NNModel, get_shape_preserving_layers_from_net
from modules import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer

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
        for offset in (-10.0, 10.0)
    ],
]

def analyze_module_impact(
    model: NNModel,
    wav: tuple[torch.Tensor, int],
    output_dir: str | Path,
    variants: list[Variant] = DEFAULT_ANALYSIS_VARIENTS,
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
            for variant in variants:
                model.reset()
                model.model.eval()
                net = model.get_net(layer.net_type)
                if variant.operation in ("multiply", "add") and not model.layer_has_weights(net[layer.index]):
                    continue
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
                    net[layer.index] = MODULES[variant.operation](layer, **variant.parameters)
                    net[layer.index].eval()
                    reconstruction = model(wav).detach().cpu()
                    if reconstruction.shape != baseline.shape:
                        raise ValueError(f"Output shape {tuple(reconstruction.shape)} differs from baseline {tuple(baseline.shape)}")
                    if not torch.isfinite(reconstruction).all():
                        raise ValueError("Reconstruction contains non-finite samples")
                    mae = mean_absolute_error(baseline, reconstruction)
                    # auraloss expects prediction first and reference second.
                    spectral = mrstft(reconstruction, baseline)
                    if not math.isfinite(mae) or not math.isfinite(spectral):
                        raise ValueError("Comparison produced non-finite metrics")
                    row.update(mae=mae, mrstft=spectral)
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
                rows.append(row)
                # Persist after each trial so long sweeps retain partial results.
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
    ranked = sorted(rows, key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ranked)
