"""Pretty-print a tree dump from per_problem.jsonl. Owner: Person 1 (Prakhar)."""

from __future__ import annotations

import argparse

from thoughtzero.config import add_config_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.parse_args()
    raise SystemExit("not implemented yet: Person 1 (Prakhar)")


if __name__ == "__main__":
    main()
