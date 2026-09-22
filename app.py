import argparse
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".wav",
    ".aif",
    ".aiff",
    ".flac",
    ".ogg",
    ".opus",
    ".mp3",
    ".m4a",
    ".caf",
}


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        metavar="DIR",
        help=(
            "RAVE model directory. The directory is only checked for existence; "
            "its internal structure is not verified. Supplying --model-path "
            "automatically implies --type RAVE. Combining --model-path with "
            "--type Encodec is always an error."
        ),
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=("RAVE", "Encodec"),
        help=(
            "Model type. May be omitted when --model-path is supplied, in which "
            "case RAVE is selected automatically. --type Encodec cannot be used "
            "together with --model-path."
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


def add_input_option(parser: argparse.ArgumentParser, *, analyze: bool) -> None:
    if analyze:
        help_text = "Input audio file."
    else:
        help_text = "Input audio file or directory containing audio files."

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help=help_text,
    )


def add_generate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--job",
        action="append",
        metavar="JOB",
        help=("Job to execute. Repeatable, e.g. " "'--job encode --job normalize'."),
    )

    parser.add_argument(
        "--tasks",
        metavar="TASKS",
        help=(
            "Tasks as a single string. Alternative to repeated --job, "
            "e.g. '--tasks \"encode normalize\"'."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-cli",
        description="Generate and analyze audio using RAVE or Encodec.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate audio.",
        description="Generate audio from an audio file or directory.",
    )
    add_common_options(generate)
    add_input_option(generate, analyze=False)
    add_generate_options(generate)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze an audio file.",
        description="Analyze a single audio file.",
    )
    add_common_options(analyze)
    add_input_option(analyze, analyze=True)

    return parser


def validate_and_normalize(args: argparse.Namespace) -> argparse.Namespace:
    # --model-path always implies RAVE.
    if args.model_path is not None:
        if not args.model_path.is_dir():
            raise ValueError(
                f"--model-path must be an existing directory: {args.model_path}"
            )

        if args.type == "Encodec":
            raise ValueError("--type Encodec cannot be used with --model-path")

        args.type = "RAVE"

    # Create the requested output directory.
    args.output.mkdir(parents=True, exist_ok=True)

    # Subcommand-specific output directory.
    args.output = args.output / args.command
    args.output.mkdir(parents=True, exist_ok=True)

    # Validate input according to the subcommand.
    if args.command == "generate":
        if args.input.is_file():
            if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(f"Not a supported audio file: {args.input}")
        elif not args.input.is_dir():
            raise ValueError(
                f"--input must be an audio file or directory: {args.input}"
            )

    elif args.command == "analyze":
        if not args.input.is_file():
            raise ValueError(f"--input must be an audio file: {args.input}")

        if args.input.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Not a supported audio file: {args.input}")

    return args


def generate(args: argparse.Namespace) -> None:
    print(f"type:   {args.type}")
    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"jobs:   {args.job}")
    print(f"tasks:  {args.tasks}")


def analyze(args: argparse.Namespace) -> None:
    print(f"type:   {args.type}")
    print(f"input:  {args.input}")
    print(f"output: {args.output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args = validate_and_normalize(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "generate":
        generate(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
