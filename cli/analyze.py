import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from cli.state import AnalyzeArgs, AppState, Command
from library.audio import mean_absolute_error, mrstft
from library.layers import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer
from library.model import (
    NNModel,
    get_shape_preserving_layers_from_net,
    get_weighted_layers_from_net,
)

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


DEFAULT_ANALYSIS_VARIANTS = [
    # Total applications; 1 is the unchanged control.
    *[Variant("repeat", {"repeats": n}) for n in (3, 5, 10)],
    # Change weights while keeping biases unchanged.
    *[
        Variant("multiply", {"weight_mul": factor, "bias_mul": 1.0})
        for factor in (5.0, 10.0, 20.0)
    ],
    *[
        Variant("add", {"weight_add": offset, "bias_add": 0.0})
        for offset in (0.2, -0.2, 0.5, -0.5)
    ],
]


def analyze_module_impact(
    model: NNModel,
    wav: tuple[torch.Tensor, int],
    output_dir: str | Path,
    variants: list[Variant] = DEFAULT_ANALYSIS_VARIANTS,
    nets=("encoder", "decoder"),
    plots: bool = True,
    slides: bool = True,
    max_artifacts: int | None = None,
    make_audio: bool = True,
    output_format: str = "png",
    filename: str = "",
) -> list[dict]:
    if max_artifacts is not None and (
        isinstance(max_artifacts, bool)
        or not isinstance(max_artifacts, int)
        or max_artifacts < 0
    ):
        raise ValueError("max_artifacts must be a nonnegative integer or None")
    if not variants:
        raise ValueError("Provide at least one variant")
    if output_format not in {"png", "jpg", "pdf"}:
        raise ValueError("output_format must be 'png', 'jpg', or 'pdf'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    plot_dir = output_dir / "plots"
    if make_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    if plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
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

        total_trials = sum(
            layer.layer_path in eligible[variant.operation]
            for layer in layers.values()
            for variant in variants
        )
        baseline = model(wav).detach().cpu().clone()
        sr = model.get_sample_rate()

        if make_audio:
            print(f"Saving baseline audio to {audio_dir / 'baseline.wav'}", flush=True)
            save_audio(audio_dir / "baseline.wav", baseline, sr)

        trial_number = 0
        for layer in layers.values():
            for variant in variants:
                if layer.layer_path not in eligible[variant.operation]:
                    continue
                trial_number += 1
                print(
                    f"Analyzing trial {trial_number}/{total_trials}: "
                    f"{variant.operation} {layer.layer_path}",
                    flush=True,
                )
                model.reset()
                model.model.eval()
                net = model.get_net(layer.net_type)
                stem = f"{len(rows):04d}_{layer.net_type.value}_{layer.index}_{variant.operation}"
                row = {
                    "filename": filename,
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
                    print(f"  Trial failed: {row['error']}", flush=True)
                rows.append(row)
                # Persist after each trial so long sweeps retain partial results.
                if output_dir is not None:
                    write_results(output_dir / "results.csv", rows)
    finally:
        print("Restoring the model after the analysis sweep...", flush=True)
        model.reset()
        print("Model restored; preparing requested outputs.", flush=True)
    rows.sort(
        key=lambda row: row["mae"] if row["mae"] is not None else -1, reverse=True
    )
    if artifacts:
        print(
            f"Saving requested artifacts for {len(artifacts)} selected trials...",
            flush=True,
        )
    for artifact_number, (row, stem, reconstruction) in enumerate(artifacts, start=1):
        try:
            if make_audio:
                audio_path = audio_dir / f"{stem}.wav"
                save_audio(audio_path, reconstruction, sr)
                row["audio"] = f"audio/{audio_path.name}"
                print(f"Saved audio: {audio_path}", flush=True)
            if plots:
                from library.plotting import plot_comparison

                parameters = json.loads(row["parameters"])
                operation_title = row["operation"].capitalize()
                if parameters:
                    parameter_text = ", ".join(
                        f"{name}={value}" for name, value in parameters.items()
                    )
                    operation_title += f" ({parameter_text})"
                title = (
                    "Reconstruction comparison for "
                    f"{row['filename']}\n {operation_title} {row['layer_path']}"
                )

                plot_path = plot_dir / f"{stem}.{output_format}"
                plot_comparison(
                    baseline,
                    reconstruction,
                    sr,
                    mae=row["mae"],
                    mrstft=row["mrstft"],
                    title=title,
                    save_path=str(plot_path),
                    show=False,
                )
                row["plot"] = f"plots/{plot_path.name}"
                print(f"Saved plot: {plot_path}", flush=True)
        except Exception as exc:
            row["error"] = f"Artifact export failed: {type(exc).__name__}: {exc}"
            print(f"  Artifact export failed: {row['error']}", flush=True)
    print(f"Writing analysis results to {output_dir / 'results.csv'}", flush=True)
    write_results(output_dir / "results.csv", rows)
    if slides and artifacts:
        from library.slides import write_impact_slides

        print("Generating slides...", flush=True)
        slides_path = write_impact_slides([item[0] for item in artifacts], output_dir)
        print(f"Saved slides to {slides_path}", flush=True)
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
        print(f"Starting analysis for {file.path.name}", flush=True)
        output_dir = app.output
        if len(app.files) > 1:
            output_dir = output_dir / file.path.stem

        make_artifacts = (
            app.args.make_audio or app.args.make_plots or app.args.make_slides
        )
        rows = analyze_module_impact(
            model=app.model,
            wav=file.wav,
            output_dir=output_dir,
            plots=app.args.make_plots,
            slides=app.args.make_slides,
            max_artifacts=app.args.save_depth if make_artifacts else 0,
            make_audio=app.args.make_audio,
            output_format=app.args.output_format,
            filename=file.path.name,
        )

        print(f"Finished analysis for {file.path.name}", flush=True)
        if app.args.make_heatmap:
            from cli.heatmap import render_heatmap

            mae_stem = output_dir / "heatmap_mae"
            print(f"Generating MAE heatmap for {file.path.name}...", flush=True)
            mae_output = render_heatmap(
                rows,
                "mae",
                file.path.name,
                mae_stem,
                app.args.output_format,
            )
            print(f"Saved MAE heatmap to {mae_output}")

            mrstft_stem = output_dir / "heatmap_mrstft"
            print(f"Generating MRSTFT heatmap for {file.path.name}...", flush=True)
            mrstft_output = render_heatmap(
                rows,
                "mrstft",
                file.path.name,
                mrstft_stem,
                app.args.output_format,
            )
            print(f"Saved MRSTFT heatmap to {mrstft_output}")


def analyze_heatmaps_from_csv(args: AnalyzeArgs) -> None:
    if args.read_output_csv is None:
        raise ValueError("--read-output-csv is required")

    with args.read_output_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No analysis rows found in {args.read_output_csv}")

    from cli.heatmap import render_heatmap

    output_dir = args.output / Command.ANALYZE.value
    output_dir.mkdir(parents=True, exist_ok=True)
    mae_stem = output_dir / "heatmap_mae"
    print(f"Generating MAE heatmap from {args.read_output_csv}...", flush=True)
    mae_output = render_heatmap(
        rows,
        "mae",
        args.read_output_csv.name,
        mae_stem,
        args.output_format,
    )
    print(f"Saved MAE heatmap to {mae_output}")

    mrstft_stem = output_dir / "heatmap_mrstft"
    print(f"Generating MRSTFT heatmap from {args.read_output_csv}...", flush=True)
    mrstft_output = render_heatmap(
        rows,
        "mrstft",
        args.read_output_csv.name,
        mrstft_stem,
        args.output_format,
    )
    print(f"Saved MRSTFT heatmap to {mrstft_output}")
