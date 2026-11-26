"""The command `nekaise`.

    nekaise image                      build the task image (needs Docker)
    nekaise example <folder>           write a small example store, to try the tool kit
    nekaise server --store <folder>    start the server that gives tasks and rewards
"""

import argparse
import sys


def main(argv: list | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["server"]:                                    # the server reads its own options
        from nekaise import server
        return server.main(argv[1:])
    parser = argparse.ArgumentParser(prog="nekaise", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("image", help="build the task image")
    example = commands.add_parser("example", help="write a small example store")
    example.add_argument("folder", help="the folder for the store, for example ./store")
    commands.add_parser("server", help="start the server; see: nekaise server --help")
    args = parser.parse_args(argv)
    from nekaise import image
    if args.command == "image":
        image.build()
    elif args.command == "example":
        from nekaise import example as maker
        print(f"Example store written: {maker.make(args.folder)}")


if __name__ == "__main__":
    main()
