"""Target-model token counting backed by the model's tiktoken encoding."""

from __future__ import annotations

from dataclasses import dataclass


class TokenizerUnavailableError(RuntimeError):
    """Raised when tiktoken or a supported model encoding is unavailable."""


@dataclass(frozen=True)
class TokenizerAdapter:
    model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenizerUnavailableError(
                "tiktoken is required for model-compatible token counts. Install requirements.txt."
            ) from exc
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError as exc:
            raise TokenizerUnavailableError(
                f"No tiktoken encoding is registered for {self.model!r}. Choose a supported OpenAI model."
            ) from exc
        object.__setattr__(self, "_encoding", encoding)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
