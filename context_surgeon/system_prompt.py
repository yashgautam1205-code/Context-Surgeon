"""Conservative cleanup for system prompts."""

from __future__ import annotations


def compress_system_prompt(prompt: str) -> tuple[str, bool]:
    """Remove only blank-line noise and exact duplicate lines; keep ordering intact."""
    if not prompt:
        return prompt, False

    lines = prompt.splitlines()
    kept: list[str] = []
    seen: set[str] = set()
    previous_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and not previous_blank:
                kept.append("")
            previous_blank = True
            continue
        previous_blank = False
        if stripped in seen:
            if kept and kept[-1] == "":
                kept.pop()
            continue
        seen.add(stripped)
        kept.append(line.rstrip())

    candidate = "\n".join(kept).strip()
    if len(candidate) >= len(prompt) or not candidate:
        return prompt, False
    return candidate, True