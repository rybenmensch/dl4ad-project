import argparse
import math
from pathlib import Path

from cli.state import (
    AUDIO_EXTENSIONS,
    AnalyzeArgs,
    Args,
    BendArgs,
    Command,
    CommonArgs,
    ModelType,
)


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r",
        "--rave-path",
        type=Path,
        metavar="DIR",
        help=(
            "RAVE model directory. Supplying this option automatically implies --type RAVE. "
            "Combining --rave-path with --type Encodec is always an error."
        ),
    )

    parser.add_argument(
        "-t",
        "--type",
        "--model",
        dest="model_type",
        choices=tuple(model_type.value for model_type in ModelType) + ("encodec", "rave"),
        help=(
            "Model type. May be omitted when --rave-path is supplied, in which "
            "case RAVE is selected automatically. --type Encodec cannot be used "
            "together with --rave-path."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory. Created if it does not exist.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[24_000, 48_000],
        default=48_000,
        help="EnCodec sample rate; RAVE uses its configured rate.",
    )


def add_input_option(
    parser: argparse.ArgumentParser,
    help_text: str,
) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nbform",
        description="Analyze and bend networks using RAVE or Encodec.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    bend = subparsers.add_parser(
        Command.BEND.value,
        aliases=["generate"],
        help="Bend networks.",
        description="Bend networks and generate audio from an audio file or directory.",
    )
    add_common_options(bend)
    add_input_option(bend, "Input audio file.")

    analyze = subparsers.add_parser(
        Command.ANALYZE.value,
        help="Analyze a model using an audio file.",
        description="Analyze a model using an audio file.",
    )
    add_common_options(analyze)
    analyze.add_argument(
        "-s",
        "--save-depth",
        type=int,
        metavar="N",
        help=(
            "Number of highest-impact trials to save as audio, plots, and slides. "
            "Omit to save all trials. Use 0 to save only the baseline and results."
        ),
    )
    add_input_option(analyze, "Input audio file or directory containing audio files.")

    sweep = subparsers.add_parser(
        "sweep",
        help="Run the reproducible analysis_kit sweep for one audio file.",
    )
    add_legacy_model_options(sweep)
    sweep.add_argument("--nets", nargs="+", choices=["encoder", "decoder"], default=["encoder", "decoder"])
    sweep.add_argument("--interventions", nargs="+", choices=["skip", "repeat", "multiply", "add"], default=["skip", "repeat", "multiply", "add"])
    sweep.add_argument("--add-offset", type=float, default=0.1)
    sweep.add_argument("--plots", action="store_true")
    sweep.add_argument("--slides", action="store_true")
    sweep.add_argument("--input", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run the corpus-level analysis and export raw and summary CSVs.",
    )
    add_legacy_model_options(evaluate)
    evaluate.add_argument("--nets", nargs="+", choices=["encoder", "decoder"], default=["encoder", "decoder"])
    evaluate.add_argument("--interventions", nargs="+", choices=["skip", "repeat", "multiply", "add"], default=["skip", "repeat", "multiply", "add"])
    evaluate.add_argument("--add-offset", type=float, default=0.1)
    evaluate.add_argument("--corpus-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    return parser


def add_legacy_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=["encodec", "rave"])
    parser.add_argument("--rave-path", type=Path)
    parser.add_argument(
        "--sample-rate", type=int, choices=[24_000, 48_000], default=48_000
    )


def parse_args(parser: argparse.ArgumentParser) -> Args | argparse.Namespace:
    namespace = parser.parse_args()
    if namespace.command in {"sweep", "evaluate"}:
        if namespace.model == "rave" or (
            namespace.model is None and namespace.rave_path is not None
        ):
            namespace.model = ModelType.RAVE
        else:
            namespace.model = ModelType.ENCODEC
        return namespace
    command = Command.BEND if namespace.command == "generate" else Command(namespace.command)
    if namespace.model_type is None:
        model_type = None
    elif namespace.model_type.lower() == "rave":
        model_type = ModelType.RAVE
    else:
        model_type = ModelType.ENCODEC

    common = CommonArgs(
        command=command,
        rave_path=namespace.rave_path,
        model_type=model_type,
        output=namespace.output,
        sample_rate=namespace.sample_rate,
    )

    if command == Command.BEND:
        return BendArgs(
            **common.__dict__,
            input=namespace.input,
        )

    if command == Command.ANALYZE:
        return AnalyzeArgs(
            **common.__dict__,
            input=namespace.input,
            save_depth=namespace.save_depth,
        )

    raise RuntimeError(f"Unknown command: {command}")


def validate_and_normalize(args: Args | argparse.Namespace) -> Args | argparse.Namespace:
    if isinstance(args, argparse.Namespace):
        if args.rave_path is not None:
            if not args.rave_path.is_dir():
                raise ValueError(f"RAVE run directory not found: {args.rave_path}")
            if args.model == ModelType.ENCODEC:
                raise ValueError("--rave-path can only be used with --model rave")
            args.model = ModelType.RAVE
        if args.command == "sweep":
            if not args.input.is_file() or args.input.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError("sweep input must be a supported audio file")
        elif not args.corpus_dir.is_dir():
            raise ValueError(f"Corpus directory not found: {args.corpus_dir}")
        return args
    # --rave-path always implies RAVE.
    if args.rave_path is not None:
        if not args.rave_path.is_dir():
            raise ValueError(
                f"--rave-path must be an existing directory: {args.rave_path}"
            )

        if args.model_type == ModelType.ENCODEC:
            raise ValueError("--type Encodec cannot be used with --rave-path")

        args.model_type = ModelType.RAVE

    if args.command == Command.BEND:
        assert isinstance(args, BendArgs)

        if args.input.is_file() and args.input.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Not a supported audio file: {args.input}")

        if not args.input.is_file() and not args.input.is_dir():
            raise ValueError(
                f"--input must be an audio file or directory: {args.input}"
            )

    elif args.command == Command.ANALYZE:
        assert isinstance(args, AnalyzeArgs)

        if args.seconds is not None and (
            not math.isfinite(args.seconds) or args.seconds <= 0
        ):
            raise ValueError("--seconds must be positive and finite")

        if args.save_depth is not None and args.save_depth < 0:
            raise ValueError("--save-depth must be a nonnegative integer")

        if args.input.is_file():
            if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(f"Not a supported audio file: {args.input}")
        elif not args.input.is_dir():
            raise ValueError(f"--input must be an audio file or directory: {args.input}")

    return args
