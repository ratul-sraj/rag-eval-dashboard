"""``python -m evalkit`` dispatcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m evalkit <command> [options]\n"
              "commands: run | calibrate | gate")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        from .runner import main as m
    elif cmd == "calibrate":
        from .calibrate import main as m
    elif cmd == "gate":
        from .gate import main as m
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    return m(rest)


if __name__ == "__main__":
    raise SystemExit(main())
