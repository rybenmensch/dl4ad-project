import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torchaudio

from analysis_kit import analyze_module_impact
from cli.state import create_model, list_audio_files
from stats import aggregate_results


def evaluate_corpus(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_path = Path(args.corpus_dir)
    output_path = Path(args.output)
    audio_files = list_audio_files(corpus_path)
    model = create_model(args.model, args.rave_path, args.sample_rate)
    interventions = tuple(args.interventions)
    model_name = args.model.value

    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_path": str(args.rave_path.resolve()) if args.rave_path else None,
        "model_sample_rate": model.get_sample_rate(),
        "model_channels": model.get_channels(),
        "corpus_dir": str(corpus_path.resolve()),
        "audio_files": [path.name for path in audio_files],
        "nets": list(args.nets),
        "interventions": list(interventions),
        "parameters": {
            "repeat_counts": [3, 5, 10] if "repeat" in interventions else [],
            "weight_multipliers": [1.0, 5.0, 10.0, 20.0]
            if "multiply" in interventions
            else [],
            "add_offset": args.add_offset if "add" in interventions else None,
        },
        "metrics": ["mae", "mae_normalized", "mrstft"],
        "normalization": {"target_rms": 0.1, "peak_limit": 0.99},
        "quality_thresholds": {"silence_rms": 1e-5, "clipping_ratio": 0.05},
    }
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    rows = []
    for audio_path in audio_files:
        waveform, input_sample_rate = torchaudio.load(str(audio_path))
        audio_output_dir = output_path / "artifacts" / audio_path.stem
        audio_rows = analyze_module_impact(
            model=model,
            wav=(waveform, input_sample_rate),
            output_dir=audio_output_dir,
            nets=tuple(args.nets),
            plots=False,
            slides=False,
            interventions=interventions,
            add_offset=args.add_offset,
            save_baseline=False,
            save_trial_audio=False,
            write_intermediate_results=False,
            write_results_csv=False,
        )
        for result in audio_rows:
            result["audio_file"] = audio_path.name
            result["layer"] = result.get("layer_path")
            result["raw_mae"] = result.get("mae")
            rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(output_path / "raw_results.csv", index=False)
    summary = aggregate_results(results, output_dir=output_path)
    print(
        f"Completed {len(results)} trials across {len(audio_files)} files. "
        f"Results: {output_path}"
    )
    return results, summary