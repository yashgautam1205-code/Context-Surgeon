"""A deliberately separate experiment; hypothetical pieces are not model tokens."""

from __future__ import annotations

import re


def _bigram_piece_count(text: str) -> int:
    words = re.findall(r"\S+", text)
    return sum((len(word) + 1) // 2 for word in words)


def run_tokenization_experiment(text: str, count_tokens) -> dict:
    """Compare target BPE count to hypothetical two-character segmentation."""
    target_count = count_tokens(text)
    hypothetical_count = _bigram_piece_count(text)
    return {
        "target_model_token_count": target_count,
        "hypothetical_two_character_piece_count": hypothetical_count,
        "hypothetical_difference": target_count - hypothetical_count,
        "compatible_with_target_model": False,
        "warning": "Hypothetical segmentation only; these pieces cannot be sent as target-model token IDs or assumed to improve model efficiency.",
    }