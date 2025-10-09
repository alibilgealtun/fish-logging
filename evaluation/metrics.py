from __future__ import annotations
"""Metric computations for ASR numeric evaluation.

All functions are pure and operate on already-normalized inputs.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional
import math

try:  # optional dependency already in requirements
    import Levenshtein  # type: ignore
except Exception:  # pragma: no cover - fallback pure python
    Levenshtein = None  # type: ignore


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if Levenshtein is not None:
        return int(Levenshtein.distance(a, b))  # type: ignore[attr-defined]
    # Simple DP fallback
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            cb = b[j - 1]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _edit_distance_tokens(hyp_tokens: List[str], ref_tokens: List[str]) -> int:
    """Classic Wagner-Fischer DP over tokens (insertions, deletions, substitutions all cost 1)."""
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp_tokens[i - 1] == ref_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def word_error_rate(hyp: str, ref: str) -> float:
    """Compute WER given hypothesis and reference strings.

    WER = (S + D + I) / N, computed via token-level edit distance.
    Tokenization uses simple whitespace split which is adequate for numeric tokens.
    Returns 0.0 if reference is empty (defined as perfect if hypothesis also empty, else 1.0).
    """
    ref_tokens = ref.strip().split()
    hyp_tokens = hyp.strip().split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    dist = _edit_distance_tokens(hyp_tokens, ref_tokens)
    return dist / max(len(ref_tokens), 1)


def char_error_rate(hyp: str, ref: str) -> float:
    ref_clean = ref.replace(" ", "")
    hyp_clean = hyp.replace(" ", "")
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = _levenshtein(hyp_clean, ref_clean)
    return dist / max(len(ref_clean), 1)


def _number_to_digit_string(x: Optional[float]) -> str:
    if x is None:
        return ""
    # Canonicalize float: trim trailing zeros and dot; then keep only digits
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "".join(ch for ch in s if ch.isdigit())


def digit_error_rate(pred_number: Optional[float], ref_number: Optional[float]) -> float:
    """Digit error rate between two numeric values based on digit sequences.
    Ignores decimal point and non-digits; compares canonicalized digits via Levenshtein.
    If reference is None returns 0 if pred is also None else 1.
    """
    if ref_number is None:
        return 0.0 if pred_number is None else 1.0
    ref_digits = _number_to_digit_string(ref_number)
    pred_digits = _number_to_digit_string(pred_number)
    if not ref_digits:
        return 0.0 if not pred_digits else 1.0
    if Levenshtein is not None:
        dist = int(Levenshtein.distance(pred_digits, ref_digits))  # type: ignore[attr-defined]
    else:
        dist = _levenshtein(pred_digits, ref_digits)
    return dist / len(ref_digits)


def numeric_exact_match(pred_number: Optional[float], ref_number: Optional[float], tolerance: float = 1e-6) -> int:
    if pred_number is None or ref_number is None:
        return 1 if pred_number is None and ref_number is None else 0
    return 1 if abs(pred_number - ref_number) <= tolerance else 0


def mean_absolute_error_numbers(pairs: Iterable[tuple[Optional[float], Optional[float]]]) -> float:
    errors: List[float] = []
    for pred, ref in pairs:
        if pred is None or ref is None:
            continue
        errors.append(abs(pred - ref))
    if not errors:
        return math.nan
    return sum(errors) / len(errors)


@dataclass
class Percentiles:
    p50: float
    p95: float
    p99: float


def compute_percentiles(values: List[float]) -> Percentiles:
    if not values:
        return Percentiles(float("nan"), float("nan"), float("nan"))
    vs = sorted(values)
    def _pct(p: float) -> float:
        if not vs:
            return float("nan")
        k = (len(vs) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vs[int(k)]
        d0 = vs[f] * (c - k)
        d1 = vs[c] * (k - f)
        return d0 + d1
    return Percentiles(p50=_pct(0.5), p95=_pct(0.95), p99=_pct(0.99))
