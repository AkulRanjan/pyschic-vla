"""Plots from one or more results folders (SPEC.md §8.3). Owner: Person 4 (Harjas)."""

from __future__ import annotations

import argparse

from thoughtzero.config import add_config_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.parse_args()
    raise SystemExit("not implemented yet: Person 4 (Harjas)")


if __name__ == "__main__":
    main()
