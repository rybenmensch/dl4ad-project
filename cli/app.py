import argparse

from cli.analyze import analyze_loop
from cli.bend import bend_loop
from cli.evaluate import evaluate_corpus
from cli.parser import build_parser, parse_args, validate_and_normalize
from cli.sweep import analyze_file
from cli.state import AppState, Command


def main() -> None:
    parser = build_parser()

    try:
        args = parse_args(parser)
        args = validate_and_normalize(args)
    except ValueError as exc:
        parser.error(str(exc))

    if isinstance(args, argparse.Namespace):
        if args.command == "sweep":
            analyze_file(args)
        else:
            evaluate_corpus(args)
        return

    app = AppState(args)

    if args.command == Command.BEND:
        bend_loop(app)
    elif args.command == Command.ANALYZE:
        analyze_loop(app)


if __name__ == "__main__":
    main()
