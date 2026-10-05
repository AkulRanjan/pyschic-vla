"""sha256-keyed disk cache for Jev responses (SPEC.md §5.4). Owner: Person 2 (Jagriti).

CLI: ``python -m thoughtzero.judge.cache export|import <file.jsonl>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import diskcache

DEFAULT_CACHE_DIR = os.environ.get("JEV_CACHE_DIR", ".cache/jev")


def cache_key(state: str, questions: dict[str, Any], model: str, prompt_version: str) -> str:
    """Transport is deliberately excluded: a direct-API hit and an OpenRouter
    hit are interchangeable as long as the model version matches."""
    payload = {
        "request": {"state": state, "questions": questions},
        "model": model,
        "prompt_version": prompt_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str = DEFAULT_CACHE_DIR) -> None:
        self.directory = directory
        self._cache = diskcache.Cache(directory)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._cache.get(key)
        return value if value is not None else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._cache.set(key, value)

    def close(self) -> None:
        self._cache.close()

    def export(self, path: str) -> int:
        n = 0
        with open(path, "w") as f:
            for key in self._cache:
                value = self._cache.get(key)
                if value is None:
                    continue
                f.write(json.dumps({"key": key, "value": value}) + "\n")
                n += 1
        return n

    def import_(self, path: str) -> int:
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["key"] in self._cache:
                    continue
                self._cache.set(row["key"], row["value"])
                n += 1
        return n


def _cli() -> None:
    parser = argparse.ArgumentParser(prog="python -m thoughtzero.judge.cache")
    sub = parser.add_subparsers(dest="cmd", required=True)
    export_p = sub.add_parser("export")
    export_p.add_argument("path")
    import_p = sub.add_parser("import")
    import_p.add_argument("path")
    args = parser.parse_args()

    cache = DiskCache()
    if args.cmd == "export":
        n = cache.export(args.path)
        print(f"exported {n} entries to {args.path}")
    elif args.cmd == "import":
        n = cache.import_(args.path)
        print(f"imported {n} new entries from {args.path}")


if __name__ == "__main__":
    _cli()
