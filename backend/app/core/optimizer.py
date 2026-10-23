"""
backend/app/core/optimizer.py

Analyzes input prompts and applies token compression strategies:
1. Strips conversational politeness/filler phrases (e.g., 'Can you please', 'I would like you to').
2. Collapses redundant whitespace and formatting overhead.
3. Normalizes repetitive punctuation and delimiters.
"""

import re
from typing import List, Tuple

FILLER_PATTERNS = [
    r"\b(please|kindly|could you please|can you please|would you be so kind as to)\b",
    r"\b(i want you to|i would like you to|your job is to|you are tasked with)\b",
    r"\b(in order to|as a matter of fact|due to the fact that)\b",
    r"\b(feel free to|make sure to|be sure to)\b",
    r"\b(it is important to note that|keep in mind that)\b",
]

REPLACEMENTS = [
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bas well as\b", "and"),
    (r"\butilize\b", "use"),
]


def optimize_prompt(text: str) -> Tuple[str, List[str]]:
    """
    Returns the compressed prompt and a log of optimizations applied.
    """
    if not text.strip():
        return text, []

    optimized = text
    applied_rules = []

    # 1. Remove polite filler prefixes
    for pat in FILLER_PATTERNS:
        match = re.search(pat, optimized, flags=re.IGNORECASE)
        if match:
            optimized = re.sub(pat, "", optimized, flags=re.IGNORECASE)
            applied_rules.append(f"Removed conversational filler: '{match.group(0)}'")

    # 2. Shorthand replacements
    for pat, rep in REPLACEMENTS:
        if re.search(pat, optimized, flags=re.IGNORECASE):
            optimized = re.sub(pat, rep, optimized, flags=re.IGNORECASE)
            applied_rules.append(f"Simplified verbose phrase to '{rep}'")

    # 3. Collapse multiple spaces and linebreaks
    lines = [line.strip() for line in optimized.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    if len(cleaned) < len(optimized):
        applied_rules.append("Collapsed redundant whitespace and blank lines")

    optimized = cleaned.strip()

    return optimized, applied_rules