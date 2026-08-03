"""Pure ranking / flip-detection helpers for the EQS-02 shape re-ranking campaign.

No solver imports: this module is about orderings only, so it is unit-testable
without running any physics (test_rank_utils.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Number = Optional[float]


def rank(values: Dict[str, float], ascending: bool = True) -> List[str]:
    """Shape names ordered by value. Ties broken by name, so the order is
    deterministic and two arms with identical ties still compare equal."""
    sign = 1.0 if ascending else -1.0
    return [k for k, _ in sorted(values.items(), key=lambda kv: (sign * kv[1], kv[0]))]


def pair_inversions(order_a: List[str], order_b: List[str]) -> List[Tuple[str, str]]:
    """Every unordered pair whose relative order differs between the two rankings,
    named in `order_a`'s orientation (first element is the one that came first in a)."""
    pos_b = {k: i for i, k in enumerate(order_b)}
    out: List[Tuple[str, str]] = []
    for i, x in enumerate(order_a):
        for y in order_a[i + 1:]:
            if pos_b[x] > pos_b[y]:
                out.append((x, y))
    return out


def spearman(order_a: List[str], order_b: List[str]) -> float:
    """Spearman rank correlation between two orderings of the same set.
    Computed from ranks directly (they are a permutation of 0..n-1, no ties)."""
    n = len(order_a)
    if n < 2:
        return float("nan")
    pos_a = {k: i for i, k in enumerate(order_a)}
    pos_b = {k: i for i, k in enumerate(order_b)}
    d2 = sum((pos_a[k] - pos_b[k]) ** 2 for k in pos_a)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def compare_rankings(legacy: Dict[str, Number], masked: Dict[str, Number],
                     ascending: bool = True) -> dict:
    """Rank the same shape set under both arms and report what moved.

    Shapes whose value is missing (None/NaN) in EITHER arm are dropped from the
    ranking and listed under "skipped" -- a failed march must not silently take a
    rank position."""
    if set(legacy) != set(masked):
        raise ValueError(f"shape sets differ: {sorted(legacy)} vs {sorted(masked)}")

    def ok(v: Number) -> bool:
        return v is not None and v == v            # not None, not NaN

    usable = sorted(k for k in legacy if ok(legacy[k]) and ok(masked[k]))
    skipped = sorted(set(legacy) - set(usable))
    lv = {k: float(legacy[k]) for k in usable}
    mv = {k: float(masked[k]) for k in usable}

    rl = rank(lv, ascending=ascending)
    rm = rank(mv, ascending=ascending)
    inv = pair_inversions(rl, rm)
    return {
        "rank_legacy": rl,
        "rank_masked": rm,
        "inversions": inv,
        "flipped": bool(inv),
        "spearman": spearman(rl, rm) if len(usable) > 1 else float("nan"),
        "skipped": skipped,
        "ascending": ascending,
        "values": {k: {"legacy": lv[k], "masked": mv[k],
                       "delta": mv[k] - lv[k],
                       "rel": (mv[k] - lv[k]) / lv[k] if lv[k] != 0 else float("nan")}
                   for k in usable},
    }
