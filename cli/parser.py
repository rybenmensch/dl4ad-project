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
            "RAVE model directory. "
            "Supplying this option automatically implies --type RAVE. "
            "Combining --rave-path with --type Encodec is always an error."
        ),
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="model_type",
        choices=tuple(model_type.value for model_type in ModelType),
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


def add_input_option(
    parser: argparse.ArgumentParser,
    help_text: str,
    required: bool = True,
) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=required,
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
        help="Bend networks.",
        description="Bend networks and generate audio from an audio file or directory.",
    )
    add_common_options(bend)
    add_input_option(bend, "Input audio file.")

    analyze = subparsers.add_parser(
        Command.ANALYZE.value,
        help="Analyze a model using audio files.",
        description="Analyze a model using an audio file or a directory of audio files.",
    )
    add_common_options(analyze)
    analyze.add_argument(
        "-s",
        "--save-depth",
        type=int,
        metavar="N",
        help=(
            "Number of highest-impact trials to save as requested artifacts. "
            "Omit to save all trials. Use 0 to save only requested baseline audio "
            "and results."
        ),
    )
    add_input_option(
        analyze,
        "Input audio file or directory containing audio files.",
        required=False,
    )
    analyze.add_argument(
        "--read-output-csv",
        type=Path,
        metavar="CSV",
        help="Read saved analysis results from a CSV instead of running a model sweep.",
    )
    analyze.add_argument(
        "--make-plots",
        action="store_true",
        help="Save baseline and reconstruction comparison plots.",
    )
    analyze.add_argument(
        "--make-slides",
        action="store_true",
        help="Save a Quarto impact slide deck.",
    )
    analyze.add_argument(
        "--make-audio",
        action="store_true",
        help="Save baseline and selected reconstruction audio.",
    )
    analyze.add_argument(
        "--make-heatmap",
        action="store_true",
        help="Save both MAE and MRSTFT layer impact heatmaps.",
    )
    analyze.add_argument(
        "--format",
        dest="output_format",
        choices=("png", "jpg", "pdf"),
        default="png",
        help="Format for plots and heatmaps (default: png).",
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> Args:
    namespace = parser.parse_args()
    command = Command(namespace.command)

    common = CommonArgs(
        command=command,
        rave_path=namespace.rave_path,
        model_type=(
            ModelType(namespace.model_type)
            if namespace.model_type is not None
            else None
        ),
        output=namespace.output,
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
            make_plots=namespace.make_plots,
            make_slides=namespace.make_slides,
            make_audio=namespace.make_audio,
            make_heatmap=namespace.make_heatmap,
            read_output_csv=namespace.read_output_csv,
            output_format=namespace.output_format,
        )

    raise RuntimeError(f"Unknown command: {command}")


def validate_and_normalize(args: Args) -> Args:
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

        if args.read_output_csv is not None:
            if args.input is not None:
                raise ValueError("--input cannot be used with --read-output-csv")
            if not args.read_output_csv.is_file():
                raise ValueError(
                    f"--read-output-csv must be an existing file: {args.read_output_csv}"
                )
            if not args.make_heatmap:
                raise ValueError("--read-output-csv requires --make-heatmap")
            if args.make_plots or args.make_slides or args.make_audio:
                raise ValueError(
                    "--read-output-csv cannot be combined with --make-plots, "
                    "--make-slides, or --make-audio"
                )
        else:
            if args.input is None:
                raise ValueError(
                    "--input is required unless --read-output-csv is provided"
                )
            if not args.input.exists():
                raise FileNotFoundError(f"Input path not found: {args.input}")

            if args.input.is_file():
                if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
                    raise ValueError(f"Not a supported audio file: {args.input}")
            elif not args.input.is_dir():
                raise ValueError(
                    f"--input must be an audio file or directory: {args.input}"
                )

    return args
