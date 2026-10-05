"""Standalone HTML page of a search tree (``dump_tree`` dict), for demos. No JavaScript needed.

Every node is a collapsible ``<details>`` element showing its step, visits N, mean value Q,
prior P and judge value; children are ordered by visits. The most-visited path from the root
(what ``most_visited`` extraction follows) is highlighted and open by default.
"""

from __future__ import annotations

import html
from typing import Any

STYLE = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --path:#e6f4ea;
        --pathline:#1a7f37; --term:#fff8c5; --bad:#ffebe9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#30363d; --path:#12261e;
          --pathline:#3fb950; --term:#2e2a14; --bad:#2d1416; }
}
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif;
       margin:0 auto; max-width:1100px; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; }
.meta { color:var(--muted); margin-bottom:16px; }
.box { border:1px solid var(--line); border-radius:8px; padding:12px 16px; margin:12px 0; }
.answer { font-size:18px; font-weight:600; }
details { margin-left:18px; border-left:2px solid var(--line); padding-left:10px; }
details.root { margin-left:0; border-left:none; padding-left:0; }
details.path > summary { background:var(--path); border-left:3px solid var(--pathline); }
summary { cursor:pointer; padding:4px 8px; border-radius:6px; list-style-position:outside; }
summary:hover { outline:1px solid var(--line); }
.stats { font:12px ui-monospace,monospace; color:var(--muted); white-space:nowrap; }
.step { white-space:pre-wrap; }
.term { background:var(--term); border-radius:4px; padding:0 6px; font-weight:600; }
.legend span { margin-right:14px; }
"""


def most_visited_ids(dump: dict[str, Any]) -> set[int]:
    """Node ids on the most-visited path from the root (ties: higher prior)."""
    ids, node = {dump["id"]}, dump
    while node.get("children"):
        node = max(node["children"], key=lambda c: (c["N"], c["prior"]))
        ids.add(node["id"])
    return ids


def _node(node: dict[str, Any], on_path: set[int], is_root: bool) -> str:
    cls = ["root"] if is_root else []
    if node["id"] in on_path:
        cls.append("path")
    value = "–" if node.get("value") is None else f"{node['value']:.2f}"
    stats = f"N={node['N']} · Q={node['Q']:.2f}"
    if not is_root:
        stats += f" · P={node['prior']:.2f}"
    stats += f" · v={value}"
    step = html.escape(node.get("step") or "(problem)")
    term = ""
    if node.get("terminal"):
        answer = node.get("final_answer")
        term = (
            f' <span class="term">answer: {html.escape(str(answer)) if answer else "none"}</span>'
        )
    children = sorted(node.get("children", []), key=lambda c: (-c["N"], -c["prior"]))
    inner = "".join(_node(c, on_path, False) for c in children)
    open_attr = " open" if node["id"] in on_path else ""
    return (
        f'<details class="{" ".join(cls)}"{open_attr}><summary><span class="stats">{stats}</span> '
        f'<span class="step">{step}</span>{term}</summary>{inner}</details>'
    )


def tree_to_html(
    dump: dict[str, Any],
    *,
    problem: str,
    answer: str | None,
    title: str = "ThoughtZero search tree",
    gold: str | None = None,
    notes: list[str] | None = None,
) -> str:
    """A complete HTML document for ``dump`` (as produced by ``search.mcts.dump_tree``)."""
    verdict = ""
    if gold is not None:
        verdict = " ✓ correct" if answer is not None and answer == gold else f" (expected {gold})"
    note_html = "".join(f"<div>{html.escape(n)}</div>" for n in notes or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{STYLE}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">{note_html}</div>
<div class="box"><b>Problem.</b> {html.escape(problem)}</div>
<div class="box answer">Answer: {html.escape(str(answer))}{html.escape(verdict)}</div>
<div class="box legend"><span><b>N</b> visits</span><span><b>Q</b> mean value</span>
<span><b>P</b> judge prior</span><span><b>v</b> judge value</span>
<span>green = most-visited path</span></div>
{_node(dump, most_visited_ids(dump), True)}
</body></html>
"""
