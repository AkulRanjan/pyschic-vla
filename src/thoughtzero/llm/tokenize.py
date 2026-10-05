"""HF tokenizer wrapper: chat template and token counting (team file §A2).

``transformers`` is imported lazily, so the rest of the package (and the
tests) work without it.
"""

from __future__ import annotations

from typing import Any, Protocol


class ChatTokenizer(Protocol):
    """What the generator needs from a tokenizer."""

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class HFChatTokenizer:
    """Wraps ``transformers.AutoTokenizer`` for one model.

    ``template_kwargs`` are forwarded to ``apply_chat_template``. By default it
    passes ``enable_thinking=False``: Gemma 4's template only enters thinking
    mode (a ``<|think|>`` token in the system turn) when that flag is true
    (spec §5.2; docs/verified_apis.md).

    ``strip_bos``: the template renders ``<bos>`` as text. Gemma 4's HF
    tokenizer does NOT add BOS on encode (verified 2026-10-05), so servers that
    tokenize with it (vLLM, SGLang) need the template's BOS kept: default False.
    Set True only for a server that adds its own BOS, to avoid doubling it.
    """

    def __init__(
        self,
        model_name: str,
        template_kwargs: dict[str, Any] | None = None,
        strip_bos: bool = False,
        **from_pretrained_kwargs: Any,
    ) -> None:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.template_kwargs = (
            {"enable_thinking": False} if template_kwargs is None else dict(template_kwargs)
        )
        self.strip_bos = strip_bos
        self._tok = AutoTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        out = str(
            self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
            )
        )
        bos = self._tok.bos_token
        if self.strip_bos and bos and out.startswith(bos):
            out = out[len(bos) :]
        return out

    def count_tokens(self, text: str) -> int:
        # Special tokens in the text (e.g. "<bos>", "<|turn>") are counted as
        # the single tokens they are; none are added.
        return len(self._tok.encode(text, add_special_tokens=False))


class ApproxTokenizer:
    """Network-free fallback for local dev and tests: ~4 chars per token.

    Its chat template mimics Gemma 4's turn format. Never use it for real experiments:
    completion tokens are the paper's primary compute axis.
    """

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        parts = ["<bos>"]
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            parts.append(f"<|turn>{role}\n{m['content'].strip()}<turn|>\n")
        parts.append("<|turn>model\n")
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        return (len(text) + 3) // 4
