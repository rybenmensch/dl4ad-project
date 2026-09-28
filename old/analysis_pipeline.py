import argparse
import json
from datetime import datetime
from pathlib import Path

import torch._dynamo
import pandas as pd
import torchaudio

from analysis_kit import analyze_module_impact
from encodec_lib import EncodecNNModel, raw_encodec_model
from stats import aggregate_results


def write_run_metadata(
    output_dir: Path,
    corpus_dir: Path,
    model_name: str,
    sample_rate: int,
    interventions: list[str],
    nets: tuple[str, ...],
    add_offset: float,
) -> None:
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_sample_rate": sample_rate,
        "corpus_dir": str(corpus_dir.resolve()),
        "audio_files": sorted(path.name for path in corpus_dir.iterdir() if path.suffix.lower() == ".wav"),
        "nets": list(nets),
        "interventions": interventions,
        "parameters": {
            "repeat_counts": [3, 5, 10] if "repeat" in interventions else [],
            "weight_multipliers": [1.0, 5.0, 10.0, 20.0] if "multiply" in interventions else [],
            "add_offset": add_offset if "add" in interventions else None,
        },
        "metrics": ["mae", "mae_normalized", "mrstft"],
        "normalization": {
            "target_rms": 0.1,
            "peak_limit": 0.99,
        },
        "quality_thresholds": {
            "silence_rms": 1e-5,
            "clipping_ratio": 0.05,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def iter_audio_files(corpus_dir: str | Path) -> list[Path]:
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    audio_files = sorted(
        (path for path in corpus_path.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.name.lower(),
    )
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in '{corpus_path}'.")

    return audio_files


def run_corpus_evaluation(
    corpus_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "encodec",
    interventions: list[str] | None = None,
    sample_rate: int = 48_000,
    nets: tuple[str, ...] = ("encoder", "decoder"),
    add_offset: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_name.lower() != "encodec":
        raise ValueError(f"Unsupported model '{model_name}'. This pipeline currently supports EnCodec only.")

    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    audio_files = iter_audio_files(corpus_path)
    interventions = interventions or ["skip", "repeat", "multiply", "add"]

    model = EncodecNNModel(raw_encodec_model(sample_rate))
    write_run_metadata(
        output_path,
        corpus_path,
        model_name,
        sample_rate,
        interventions,
        nets,
        add_offset,
    )

    rows = []
    for audio_path in audio_files:
        waveform, input_sample_rate = torchaudio.load(str(audio_path))
        audio_output_dir = output_path / "artifacts" / audio_path.stem
        audio_rows = analyze_module_impact(
            model=model,
            wav=(waveform, input_sample_rate),
            output_dir=audio_output_dir,
            nets=nets,
            plots=False,
            slides=False,
            interventions=tuple(interventions),
            add_offset=add_offset,
        )

        for result in audio_rows:
            result["audio_file"] = audio_path.name
            result["layer"] = result.get("layer_path")
            result["raw_mae"] = result.get("mae")
            rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(output_path / "raw_results.csv", index=False)
    summary = aggregate_results(results, output_dir=output_path)
    return results, summary


run_evaluation_pipeline = run_corpus_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corpus-level EnCodec layer-intervention evaluation.")
    parser.add_argument("--corpus-dir", default="audio/source", help="Directory containing WAV files.")
    parser.add_argument("--output-dir", default="results", help="Where per-file artifacts and CSVs are written.")
    parser.add_argument("--model-name", default="encodec", choices=["encodec"])
    parser.add_argument("--sample-rate", type=int, choices=[24_000, 48_000], default=48_000)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=["skip", "repeat", "multiply", "add"],
        default=["skip", "repeat", "multiply", "add"],
    )
    parser.add_argument("--add-offset", type=float, default=0.1)
    parser.add_argument(
        "--nets",
        nargs="+",
        choices=["encoder", "decoder"],
        default=["encoder", "decoder"],
    )
    args = parser.parse_args()

    run_corpus_evaluation(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        interventions=args.interventions,
        sample_rate=args.sample_rate,
        nets=tuple(args.nets),
        add_offset=args.add_offset,
    )


if __name__ == "__main__":
    main()
