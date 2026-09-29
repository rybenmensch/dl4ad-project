from pathlib import Path

import torchaudio

from analysis_kit import analyze_module_impact
from cli.state import create_model, list_audio_files


def analyze_file(args) -> list[dict]:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Sweep expects one audio file: {input_path}")
    audio_path = list_audio_files(input_path, allow_directory=False)[0]
    waveform, sample_rate = torchaudio.load(str(audio_path))
    model = create_model(args.model, args.rave_path, args.sample_rate)
    results = analyze_module_impact(
        model=model,
        wav=(waveform, sample_rate),
        output_dir=args.output,
        nets=tuple(args.nets),
        plots=args.plots,
        slides=args.slides,
        interventions=tuple(args.interventions),
        add_offset=args.add_offset,
    )
    print(f"Completed {len(results)} trials. Results: {args.output}")
    return results