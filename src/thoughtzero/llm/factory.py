"""Build the generator the config asks for (``generator.api``)."""

from __future__ import annotations

from typing import Any

from thoughtzero.config import Config
from thoughtzero.types import Generator


def make_generator(cfg: Config, **kwargs: Any) -> Generator:
    """``completions``: ``OpenAICompatibleGenerator`` (self-hosted server); ``chat``:
    ``ChatGenerator`` (hosted chat APIs). ``kwargs`` go to the class (e.g. ``tokenizer_name``).
    """
    if cfg.generator.api == "chat":
        from thoughtzero.llm.chat_generator import ChatGenerator

        return ChatGenerator.from_config(cfg, **kwargs)
    from thoughtzero.llm.gemma import OpenAICompatibleGenerator

    return OpenAICompatibleGenerator.from_config(cfg, **kwargs)
