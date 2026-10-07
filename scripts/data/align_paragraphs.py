"""Re-align DGT-Acquis paragraphs within a document.

The DGT-Acquis language files are aligned by line number, but in many
documents the German (and some French) paragraphs are shifted by one or more
lines relative to English, and the shift can change inside a document. Each
English paragraph is therefore matched to a paragraph of the other language
within a small window, using two signals:

- length: the log character-length ratio of true translations is close to a
  language-specific mean with a small spread;
- numbers: legal paragraphs usually carry the same numbers (article,
  regulation and amount references) in every language.

A match is accepted only if it is good enough, clearly better than the
runner-up, and mutual (the other paragraph's best English match is the same
paragraph). Ambiguous paragraphs are dropped rather than guessed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NUMBER_RE = re.compile(r"\d+(?:[.,/\-]\d+)*")

# Mean and spread of log((len(other) + 1) / (len(en) + 1)) for correct pairs,
# measured on DGT-Acquis paragraphs whose numbers agree exactly.
LENGTH_MODEL = {"fr": (0.062, 0.12), "de": (0.066, 0.18)}
NUMBER_WEIGHT = 4.0
WINDOW = 5
MAX_COST = 4.5
MIN_MARGIN = 2.0


@dataclass(frozen=True)
class Paragraph:
    text: str
    length: int
    numbers: frozenset[str]


def paragraph(text: str) -> Paragraph:
    numbers = frozenset(
        digits
        for match in NUMBER_RE.findall(text)
        for digits in [re.sub(r"\D", "", match)]
        if len(digits) >= 2
    )
    return Paragraph(text, len(text), numbers)


def match_cost(english: Paragraph, other: Paragraph, language: str) -> float:
    """Lower is better; about 0 for a typical correct translation pair."""
    if not english.text or not other.text:
        return math.inf
    mean, spread = LENGTH_MODEL[language]
    z = (math.log((other.length + 1) / (english.length + 1)) - mean) / spread
    cost = 0.5 * z * z
    if english.numbers or other.numbers:
        union = english.numbers | other.numbers
        cost += NUMBER_WEIGHT * (1 - len(english.numbers & other.numbers) / len(union))
    return cost


def best_two(costs: list[tuple[float, int]]) -> tuple[float, int, float]:
    ranked = sorted(costs)
    best_cost, best_index = ranked[0] if ranked else (math.inf, -1)
    second = ranked[1][0] if len(ranked) > 1 else math.inf
    return best_cost, best_index, second


def align_paragraph(
    english: list[Paragraph], other: list[Paragraph], index: int, language: str
) -> tuple[int, float] | None:
    """Index of the paragraph in `other` that translates english[index], or None."""
    window = range(max(0, index - WINDOW), min(len(other), index + WINDOW + 1))
    cost, match, runner_up = best_two(
        [(match_cost(english[index], other[j], language), j) for j in window]
    )
    if match < 0 or cost > MAX_COST or runner_up - cost < MIN_MARGIN:
        return None
    # Mutual check: the chosen paragraph must prefer this English paragraph.
    back_window = range(max(0, match - WINDOW), min(len(english), match + WINDOW + 1))
    _, back, _ = best_two(
        [(match_cost(english[i], other[match], language), i) for i in back_window]
    )
    if back != index:
        return None
    return match, cost
