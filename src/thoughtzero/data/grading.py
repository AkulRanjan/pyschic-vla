"""Answer extraction and equivalence (spec §8.2, team file §A4.2).

Every accuracy number in the project goes through this module, and nobody
else writes answer parsing (team file §B4, item 6). Everything here is pure
and synchronous.
"""

from __future__ import annotations

import logging
import re
import threading
from decimal import Decimal, InvalidOperation

log = logging.getLogger(__name__)

MATH_VERIFY_TIMEOUT_S = 5.0

_BOX_CMD_RE = re.compile(r"\\(?:boxed|fbox)(?![A-Za-z])")
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*(?:is)?\s*[:=]?", re.IGNORECASE)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _braced_content(text: str, open_idx: int) -> str | None:
    """Content of the ``{...}`` group opening at ``text[open_idx]``, by brace counting.

    Escaped braces (``\\{``, ``\\}``) are skipped. Returns None if unbalanced.
    """
    depth = 0
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
        i += 1
    return None


def _box_argument(text: str, cmd_end: int) -> str | None:
    """Argument of a ``\\boxed``/``\\fbox`` command whose name ends at ``cmd_end``."""
    i = cmd_end
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return None
    if text[i] == "{":
        return _braced_content(text, i)
    # Brace-less form, e.g. "\boxed 5": take the next token.
    m = re.match(r"[^\s$]+", text[i:])
    return m.group(0) if m else None


def extract_answer(text: str | None) -> str | None:
    """The final answer in ``text``: the content of the LAST ``\\boxed{...}``.

    Handles nested braces (``\\boxed{\\frac{1}{2}}``), ``\\boxed {..}``, ``\\fbox{..}``
    and nested boxes (the innermost wins, since it starts last). Falls back to
    the last ``Final answer: X`` line. Returns None if nothing is found.
    """
    if not text:
        return None
    for m in reversed(list(_BOX_CMD_RE.finditer(text))):
        arg = _box_argument(text, m.end())
        if arg is not None and arg.strip():
            return arg.strip()

    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        rest = text[matches[-1].end() :].strip().splitlines()
        if rest:
            ans = rest[0].strip().strip("*").strip()
            ans = ans.strip("$").strip().rstrip(".").strip()
            if ans:
                return ans
    return None


def has_final_answer(step: str) -> bool:
    """True if a step states a final answer (``\\boxed{``/``\\fbox`` or "Final answer")."""
    return bool(_BOX_CMD_RE.search(step) or _FINAL_ANSWER_RE.search(step))


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_UNIT_RE = re.compile(
    r"^(?P<num>.*?[\d}])\s*\\(?:text|mbox|mathrm)\{\s*[A-Za-z][A-Za-z .]*\}(?:\^\{?\d\}?)?$"
)
_TEXT_WRAP_RE = re.compile(r"\\(?:text|textbf|textit|mbox|mathrm)\{([^{}]*)\}")
_THOUSANDS_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_DECIMAL_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SLASH_FRAC_RE = re.compile(r"(-?)(\d+)/(\d+)")


def _canonical_number(s: str) -> str:
    try:
        d = Decimal(s)
    except InvalidOperation:
        return s
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def normalize_answer(ans: str | None) -> str:
    """Canonical string for an answer, used for grouping and string comparison.

    Used by ``value_vote`` and self-consistency to group answers, and as the
    fallback in :func:`is_equivalent`. ``None`` normalizes to ``""``.
    """
    if ans is None:
        return ""
    s = ans.strip()
    if s.startswith(("\\boxed", "\\fbox")):
        s = extract_answer(s) or s

    s = s.replace("\n", " ")
    for tok in ("\\!", "\\,", "\\;", "\\:", "\\ "):
        s = s.replace(tok, " " if tok == "\\ " else "")
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\displaystyle", "")
    # Degrees.
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|°|\\degree", "", s)
    # Dollars and percent signs.
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.strip()

    # Trailing text units after a number: "5 \text{ cm}", "12\text{ inches}^2".
    m = _UNIT_RE.match(s)
    if m:
        s = m.group("num")
    # Remaining \text{...} wrappers: "\text{Monday}" -> "Monday".
    s = _TEXT_WRAP_RE.sub(r"\1", s)

    s = s.strip().rstrip(".").strip()

    # A leading "x =" (single variable, single equals sign).
    lhs = re.match(r"^[A-Za-z]\s*=\s*(.+)$", s)
    if lhs and s.count("=") == 1:
        s = lhs.group(1)

    s = re.sub(r"\s+", "", s)
    s = s.replace("{,}", "")
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")

    # \sqrt2 -> \sqrt{2};  \frac12 -> \frac{1}{2}, \frac1{2}, \frac{1}2.
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)

    # "3/4" -> "\frac{3}{4}".
    slash = _SLASH_FRAC_RE.fullmatch(s)
    if slash:
        s = f"{slash.group(1)}\\frac{{{slash.group(2)}}}{{{slash.group(3)}}}"

    # ".5" / "0.50" -> "0.5"; "5.0" -> "5"; "1.5e3" -> "1500".
    if _DECIMAL_RE.fullmatch(s):
        s = _canonical_number(s)

    # Multiple-choice letters "(A)" -> "A", and plain words are case-insensitive.
    letter = re.fullmatch(r"\(([A-Za-z])\)", s)
    if letter:
        s = letter.group(1)
    if re.fullmatch(r"[A-Za-z]+", s):
        s = s.lower()
    return s


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------


def _math_verify(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    # Timeouts disabled here: math-verify uses signal.alarm (main-thread only,
    # unavailable on Windows). We enforce our own timeout around this call.
    gold_p = parse(f"${gold}$", parsing_timeout=None)
    pred_p = parse(f"${pred}$", parsing_timeout=None)
    if not gold_p or not pred_p:
        return False
    return bool(verify(gold_p, pred_p, timeout_seconds=None))


_warm_lock = threading.Lock()
_warmed = False


def _warm_up_math_verify() -> None:
    """Import math-verify and run one tiny check, once per process, outside any timeout.

    The first call loads sympy and the LaTeX parser, which can take longer than the
    per-comparison timeout on a cold machine (e.g. a fresh CI runner) and would
    otherwise make a correct answer time out.
    """
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        try:
            _math_verify("1", "1")
        except Exception as e:
            log.debug("math-verify warm-up failed: %s", e)
        _warmed = True


def _math_verify_with_timeout(pred: str, gold: str, timeout_s: float) -> bool | None:
    """Run math-verify in a daemon thread. None on timeout or error.

    sympy can hang on pathological input. A Python thread cannot be killed, so
    a hung check is abandoned (daemon) rather than stopped. That is acceptable
    for rare cases and much cheaper than a subprocess per comparison.
    """
    _warm_up_math_verify()
    result: list[bool] = []

    def run() -> None:
        try:
            result.append(_math_verify(pred, gold))
        except Exception as e:  # math-verify raises a wide range of sympy errors
            log.debug("math-verify error on %r vs %r: %s", pred, gold, e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        log.warning("math-verify timed out after %.1fs on %r vs %r", timeout_s, pred, gold)
        return None
    return result[0] if result else None


def is_equivalent(
    pred: str | None, gold: str | None, timeout_s: float = MATH_VERIFY_TIMEOUT_S
) -> bool:
    """Whether a predicted answer matches the gold answer.

    Equivalent if the normalized strings match or ``math-verify`` (with a
    timeout) says so. ``None`` is never equivalent to anything.
    """
    if pred is None or gold is None:
        return False
    n_pred, n_gold = normalize_answer(pred), normalize_answer(gold)
    if not n_pred or not n_gold:
        return False
    if n_pred == n_gold:
        return True
    return _math_verify_with_timeout(pred, gold, timeout_s) is True
