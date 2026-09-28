import argparse
from pathlib import Path

# Preload Torch's lazy imports before the first EnCodec inference on macOS.
import torch._dynamo

from cli.analyze import analyze_file
from cli.evaluate import evaluate_corpus
from cli.generate import generate_loop
from cli.state import ModelType


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", choices=[model.value for model in ModelType], default="encodec"
    )
    parser.add_argument(
        "--rave-path",
        type=Path,
        help="RAVE run directory containing config.gin and a checkpoint.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[24_000, 48_000],
        default=48_000,
        help="EnCodec model sample rate; RAVE uses the rate in its run configuration.",
    )


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nets",
        nargs="+",
        choices=["encoder", "decoder"],
        default=["encoder", "decoder"],
    )
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=["skip", "repeat", "multiply", "add"],
        default=["skip", "repeat", "multiply", "add"],
    )
    parser.add_argument("--add-offset", type=float, default=0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbform", description="Interactively bend or systematically evaluate audio models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Interactively modify a model and listen or export audio."
    )
    _add_model_arguments(generate)
    generate.add_argument("--input", type=Path, required=True, help="Audio file or directory.")
    generate.add_argument("--output", type=Path, required=True, help="Output directory for modified audio.")

    analyze = subparsers.add_parser(
        "analyze", help="Run a layer sweep for a single audio file."
    )
    _add_model_arguments(analyze)
    _add_analysis_arguments(analyze)
    analyze.add_argument("--input", type=Path, required=True, help="Single input audio file.")
    analyze.add_argument("--output", type=Path, required=True, help="Output directory for trial results.")
    analyze.add_argument("--plots", action="store_true", help="Save comparison plots.")
    analyze.add_argument("--slides", action="store_true", help="Write Quarto/reveal.js slides.")

    evaluate = subparsers.add_parser(
        "evaluate", help="Run corpus-level analysis and export raw and summary CSVs."
    )
    _add_model_arguments(evaluate)
    _add_analysis_arguments(evaluate)
    evaluate.add_argument(
        "--corpus-dir", type=Path, required=True, help="Directory containing WAV files."
    )
    evaluate.add_argument("--output", type=Path, required=True, help="Output directory for CSVs and artifacts.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.model = ModelType(args.model)

    if args.model == ModelType.RAVE and args.rave_path is None:
        parser.error("--model rave requires --rave-path")
    if args.model == ModelType.ENCODEC and args.rave_path is not None:
        parser.error("--rave-path can only be used with --model rave")

    try:
        if args.command == "generate":
            generate_loop(args)
        elif args.command == "analyze":
            analyze_file(args)
        elif args.command == "evaluate":
            evaluate_corpus(args)
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()