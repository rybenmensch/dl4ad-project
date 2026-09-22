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
        "-r",
        "--rave-path",
        type=Path,
        metavar="DIR",
        help=(
            "RAVE model directory. Supplying this option automatically implies --type RAVE."
            "Combining --model-path with --type Encodec is always an error."
        ),
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=("RAVE", "Encodec"),
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


def add_input_option(parser: argparse.ArgumentParser, help_text: str) -> None:
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
        prog="networkbend-cli",
        description="Analyze and bend networks using RAVE or Encodec.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate audio.",
        description="Generate audio from an audio file or directory.",
    )
    add_common_options(generate)
    add_input_option(generate, "Input audio file.")

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze a model using an audio file.",
        description="Analyze a model using an audio file.",
    )
    add_common_options(analyze)
    add_input_option(analyze, "Input audio file or directory containing audio files.")

    export = subparsers.add_parser(
        "export",
        help="Export bended network as torchscript.",
        description="Export bended network as torchscript.",
    )
    add_common_options(export)

    return parser


def validate_and_normalize(args: argparse.Namespace) -> argparse.Namespace:
    # --rave-path always implies RAVE.
    if args.rave_path is not None:
        if not args.rave_path.is_dir():
            raise ValueError(
                f"--rave-path must be an existing directory: {args.rave_path}"
            )

        if args.type == "Encodec":
            raise ValueError("--type Encodec cannot be used with --rave-path")

        args.type = "RAVE"

    # Create the requested output directory.
    args.output.mkdir(parents=True, exist_ok=True)

    # Subcommand-specific output directory.
    args.output = args.output / args.command
    args.output.mkdir(parents=True, exist_ok=True)

    # Validate input according to the subcommand.
    if args.command == "generate":
        if args.input.is_file() and args.input.suffix.lower() not in AUDIO_EXTENSIONS:
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


def export(args: argparse.Namespace) -> None:
    print("Not implemented yet!")


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
    elif args.command == "export":
        export(args)


if __name__ == "__main__":
    main()
