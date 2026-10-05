"""Text rendering of a ``dump_tree`` dict, for ``scripts/view_tree.py``. Owner: Person 1.

One line per node, children sorted by visit count (most visited first)::

    N=32  Q=0.591                               (root)
    ├─ N=13  Q=0.792  P=0.15  v=0.90            add 1 -> 1
    │  └─ N=6   Q=0.933  P=0.25  v=0.90  [T] 13  add 2 -> 3 ...
"""

from __future__ import annotations

from typing import Any


def _label(node: dict[str, Any], step_chars: int, is_root: bool) -> str:
    stats = f"N={node['N']:<3} Q={node['Q']:.3f}"
    if not is_root:
        stats += f"  P={node['prior']:.2f}"
    value = node.get("value")
    stats += "  v=" + ("-   " if value is None else f"{value:.2f}")
    if node.get("terminal"):
        answer = node.get("final_answer")
        stats += f"  [T] {answer if answer is not None else '(no answer)'}"
    step = " ".join(str(node.get("step", "")).split())  # one line
    if len(step) > step_chars:
        step = step[: step_chars - 3] + "..."
    return f"{stats}  {step or '(root)'}"


def format_tree(
    dump: dict[str, Any],
    *,
    max_depth: int | None = None,
    min_visits: int = 0,
    step_chars: int = 80,
) -> str:
    """Indented tree, children by descending N (ties: higher prior first).

    Children with ``N < min_visits`` and levels below ``max_depth`` are summarised in a
    single "... n hidden" line, so large trees stay readable.
    """
    lines = [_label(dump, step_chars, is_root=True)]

    def visit(node: dict[str, Any], prefix: str, depth: int) -> None:
        children = sorted(node.get("children", []), key=lambda c: (-c["N"], -c["prior"]))
        shown = [c for c in children if c["N"] >= min_visits]
        if max_depth is not None and depth >= max_depth:
            shown = []
        hidden = len(children) - len(shown)
        for i, child in enumerate(shown):
            last = i == len(shown) - 1 and not hidden
            lines.append(prefix + ("└─ " if last else "├─ ") + _label(child, step_chars, False))
            visit(child, prefix + ("   " if last else "│  "), depth + 1)
        if hidden:
            lines.append(prefix + f"└─ ... {hidden} hidden")

    visit(dump, "", 0)
    return "\n".join(lines)
