"""Disk cache for Jev calls (spec A4.3).

Key: sha256 of canonical JSON of {request, model, prompt_version}. The
transport is deliberately left out of the key, so switching transports
doesn't invalidate the cache as long as the model version matches.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import diskcache


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(state: str, questions: list[dict[str, Any]], model: str, prompt_version: str) -> str:
    payload = {
        "request": {"state": state, "questions": questions},
        "model": model,
        "prompt_version": prompt_version,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    raw_response: list[dict[str, Any]]
    parsed: dict[str, Any] | None
    input_tokens: int
    latency_ms: float
    created_at: float


def _cache_dir() -> str:
    return os.environ.get("JEV_CACHE_DIR", ".cache/jev")


class DiskCache:
    def __init__(self, directory: str | None = None) -> None:
        self._cache = diskcache.Cache(directory or _cache_dir())

    def get(self, key: str) -> CacheEntry | None:
        raw = self._cache.get(key)
        if raw is None:
            return None
        return CacheEntry(**raw)

    def set(self, key: str, entry: CacheEntry) -> None:
        self._cache.set(key, asdict(entry))

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
