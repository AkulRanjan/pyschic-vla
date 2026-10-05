"""Checks against the real Gemma 4 tokenizer files (downloads ~30 MB; skipped by default).

Run with:  pytest -m network tests/test_tokenize_network.py
"""

from __future__ import annotations

import pytest

from thoughtzero.llm.prompts import generator_prompt

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tok():  # type: ignore[no-untyped-def]
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from thoughtzero.llm.tokenize import HFChatTokenizer

    return HFChatTokenizer("google/gemma-4-E4B-it")


def test_real_template_shape(tok) -> None:  # type: ignore[no-untyped-def]
    p = generator_prompt("What is $1+1$?", ["We add."], tok)
    assert p.startswith("<bos><|turn>system\n")  # native system role, BOS kept
    assert "<|think|>" not in p  # non-thinking mode
    assert p.endswith("<turn|>\n<|turn>model\nStep 1: We add.\n\nStep 2:")


def test_tokenizer_does_not_add_bos(tok) -> None:  # type: ignore[no-untyped-def]
    # Why strip_bos defaults to False: the server won't add BOS for us.
    ids = tok._tok.encode("x")
    assert ids[0] != tok._tok.bos_token_id
