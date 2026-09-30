from cli.analyze import analyze_loop
from cli.bend import bend_loop
from cli.parser import build_parser, parse_args, validate_and_normalize
from cli.state import AppState, Command


def main() -> None:
    parser = build_parser()

    try:
        args = parse_args(parser)
        args = validate_and_normalize(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    app = AppState(args)

    if args.command == Command.BEND:
        bend_loop(app)
    elif args.command == Command.ANALYZE:
        analyze_loop(app)


if __name__ == "__main__":
    main()
