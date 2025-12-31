from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import math
import numpy as np
from scipy.special import gammaln


# Conjugate Normal-Inverse-Gamma prior for Gaussian unknown mean/variance
# Prior hyperparameters: mu0, kappa0, alpha0, beta0
# Returns log marginal likelihood p(x_{s:e}) integrated over mean and variance

def _segment_log_marginal_likelihood(x: np.ndarray, mu0: float, kappa0: float, alpha0: float, beta0: float) -> float:
    n = x.size
    if n == 0:
        return float("-inf")
    mean_x = float(x.mean())
    ssq = float(((x - mean_x) ** 2).sum())

    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2.0
    mu_diff_sq = (mean_x - mu0) ** 2
    beta_n = beta0 + 0.5 * ssq + (kappa0 * n) * mu_diff_sq / (2.0 * kappa_n)

    # Using conjugate marginal likelihood for Normal with unknown mean/variance
    # p(x) = (Gamma(alpha_n)/Gamma(alpha0)) * (beta0^alpha0 / beta_n^alpha_n) * sqrt(kappa0/kappa_n) * (pi)^(-n/2)
    log_ml = (
        gammaln(alpha_n)
        - gammaln(alpha0)
        + alpha0 * math.log(beta0)
        - alpha_n * math.log(beta_n)
        + 0.5 * (math.log(kappa0) - math.log(kappa_n))
        - (n / 2.0) * math.log(math.pi)
    )
    return float(log_ml)


def compute_bayes_factor_best_split(
    series: np.ndarray,
    start: int,
    end: int,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
    min_seg_len: int = 20,
) -> Tuple[float, Optional[int]]:
    """
    Compute maximum log Bayes factor and best split point in [start, end).
    BF = log p(left) + log p(right) - log p(whole).
    Returns (best_log_bf, best_index) with best_index None if no valid split.
    """
    n = end - start
    if n < 2 * min_seg_len:
        return 0.0, None
    x = series[start:end]
    log_ml_whole = _segment_log_marginal_likelihood(x, mu0, kappa0, alpha0, beta0)

    best_bf = 0.0
    best_k: Optional[int] = None

    for k in range(start + min_seg_len, end - min_seg_len + 1):
        left = series[start:k]
        right = series[k:end]
        log_ml_left = _segment_log_marginal_likelihood(left, mu0, kappa0, alpha0, beta0)
        log_ml_right = _segment_log_marginal_likelihood(right, mu0, kappa0, alpha0, beta0)
        bf = (log_ml_left + log_ml_right) - log_ml_whole
        if bf > best_bf:
            best_bf = bf
            best_k = k

    return float(best_bf), best_k


def bayesian_binary_segmentation(
    series: np.ndarray,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    min_seg_len: int = 20,
    bf_threshold: float = 5.0,
    max_changes: Optional[int] = None,
) -> Dict[str, List]:
    """
    Bayesian binary segmentation using marginal likelihood under NIG prior.
    - bf_threshold: threshold on log Bayes factor to accept a split (in nats).
    - max_changes: optional cap on number of change points.

    Returns dict with keys:
      - change_points: sorted indices where new segment starts (excluding 0, including end as n)
      - bayes_factors: list of BF values for accepted splits (aligned to change_points except final n)
      - segments: list of (start, end)
    """
    n = int(series.size)
    if n == 0:
        return {"change_points": [], "bayes_factors": [], "segments": []}

    # Work list of segments to test (start, end)
    stack: List[Tuple[int, int]] = [(0, n)]
    changes: List[int] = []
    bfs: List[float] = []

    while stack:
        s, e = stack.pop()
        bf, k = compute_bayes_factor_best_split(series, s, e, mu0, kappa0, alpha0, beta0, min_seg_len)
        if k is not None and bf >= bf_threshold:
            # Accept split
            changes.append(k)
            bfs.append(bf)
            # Push subsegments for further splitting
            left = (s, k)
            right = (k, e)
            # Depth-first: split the larger segment later for balance
            if right[1] - right[0] > left[1] - left[0]:
                stack.append(right)
                stack.append(left)
            else:
                stack.append(left)
                stack.append(right)
            if max_changes is not None and len(changes) >= max_changes:
                break

    changes = sorted(set(changes))

    # Build segments
    segs: List[Tuple[int, int]] = []
    prev = 0
    for cp in changes + [n]:
        segs.append((prev, cp))
        prev = cp

    return {"change_points": changes + [n], "bayes_factors": bfs, "segments": segs}


def detect_change_points(
    series: List[float] | np.ndarray,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    min_seg_len: int = 20,
    bf_threshold: float = 5.0,
    max_changes: Optional[int] = None,
) -> Dict[str, List]:
    arr = np.asarray(series, dtype=float)
    return bayesian_binary_segmentation(
        arr,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
        min_seg_len=min_seg_len,
        bf_threshold=bf_threshold,
        max_changes=max_changes,
    )
