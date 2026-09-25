from cli.analyze import analyze_loop
from cli.generate import generate_loop
from cli.parser import build_parser, parse_args, validate_and_normalize
from cli.state import AppState, Command


def main() -> None:
    parser = build_parser()

    try:
        args = parse_args(parser)
        args = validate_and_normalize(args)
    except ValueError as exc:
        parser.error(str(exc))

    app = AppState(args)

    if args.command == Command.GENERATE:
        generate_loop(app)
    elif args.command == Command.ANALYZE:
        analyze_loop(app)
    elif args.command == Command.EXPORT:
        pass


if __name__ == "__main__":
    main()
