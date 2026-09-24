from cli import build_parser, parse_args, validate_and_normalize
from state import AppState, Command


def main() -> None:
    parser = build_parser()

    try:
        args = parse_args(parser)
        args = validate_and_normalize(args)
    except ValueError as exc:
        parser.error(str(exc))

    app = AppState(args)

    if args.command == Command.GENERATE:
        pass
    elif args.command == Command.ANALYZE:
        pass
    elif args.command == Command.EXPORT:
        pass


if __name__ == "__main__":
    main()
