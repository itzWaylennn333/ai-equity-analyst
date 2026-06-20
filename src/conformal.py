"""Conformal prediction intervals (platform Layer 3b / P7 Tier 3).

Distribution-free uncertainty bands for the predictive layer -- the honest "we don't know exactly"
surface. Deterministic numpy. The headline is MARGINAL coverage, reported alongside ROLLING and
CRISIS-WINDOW coverage so the average never hides a regime where the band fails.

Verified evidence (docs/RESEARCH.md C+):
 - Split (inductive) conformal gives exact finite-sample MARGINAL coverage >= 1-alpha UNDER
   EXCHANGEABILITY (Vovk, Gammerman & Shafer 2005). Use a volatility-normalized / CQR score so band
   WIDTH scales with volatility (plain |y-yhat| gives useless constant-width bands on returns).
 - Exchangeability FAILS for return series; the coverage shortfall is bounded by drift (TV distance)
   (Barber, Candes, Ramdas & Tibshirani, Annals of Statistics 2023). Recency-weighting reduces but
   does NOT restore the guarantee.
 - Adaptive Conformal Inference restores long-run TIME-AVERAGED coverage with NO distributional
   assumption, via alpha_{t+1} = alpha_t + gamma*(alpha - err_t) (Gibbs & Candes, NeurIPS 2021).
   Per-interval coverage is NOT guaranteed (it can emit infinite or empty sets to catch up).
 - Exact CONDITIONAL coverage is impossible without infinite-width intervals (Foygel Barber et al.
   2021) -> always report local/crisis coverage, never claim a per-day/per-stock guarantee.
"""
from __future__ import annotations

import math

import numpy as np


def absolute_residual_scores(y, pred):
    """Plain nonconformity score |y - yhat| (constant-width bands; prefer the normalized score)."""
    return np.abs(np.asarray(y, float) - np.asarray(pred, float))


def normalized_scores(y, pred, sigma):
    """Volatility-normalized score |y - yhat| / sigma so interval width scales with volatility."""
    sig = np.asarray(sigma, float)
    return np.abs(np.asarray(y, float) - np.asarray(pred, float)) / np.where(sig > 0, sig, np.nan)


def split_conformal_quantile(scores, alpha: float) -> float:
    """Conformal quantile: the ceil((n+1)(1-alpha))/n empirical order statistic of calibration
    scores. Returns +inf when there are too few points to guarantee 1-alpha (cover-all), -inf when
    alpha>=1 (empty)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    if k < 1:
        return float("-inf")
    return float(s[k - 1])


def weighted_conformal_quantile(scores, alpha: float, decay: float = 0.99) -> float:
    """Recency-weighted conformal quantile (weights ~ decay^age, newest last). Robustifies against
    drift (Barber et al. 2023) but does NOT restore the exact guarantee. +inf if the mass to reach
    1-alpha sits on the (implicit) test-point atom."""
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0 or alpha <= 0.0:
        return float("inf")                                # no calib / alpha<=0 -> cover all
    if alpha >= 1.0:
        return float("-inf")                               # alpha>=1 -> empty (matches split_conformal_quantile)
    w = decay ** np.arange(n - 1, -1, -1)                  # oldest..newest -> newest weighted most
    order = np.argsort(s)
    s_sorted, w_sorted = s[order], w[order]
    total = w.sum() + 1.0                                  # +1 weight atom for the (implicit) test point
    cum = np.cumsum(w_sorted) / total
    hit = np.searchsorted(cum, 1.0 - alpha, side="left")
    return float(s_sorted[hit]) if hit < n else float("inf")


def conformal_interval(pred, q: float, sigma=1.0):
    """Interval pred +/- q*sigma (sigma=1 for absolute scores; the vol estimate for normalized)."""
    pred = np.asarray(pred, float)
    sig = np.asarray(sigma, float)
    half = q * sig
    return pred - half, pred + half


class ACI:
    """Adaptive Conformal Inference (Gibbs & Candes 2021): online update of the working miscoverage
    level so realized coverage tracks the target under arbitrary drift.
    alpha_t += gamma*(target - err_t), err_t = 1 if the last point was missed."""

    def __init__(self, alpha_target: float = 0.1, gamma: float = 0.05):
        self.target = alpha_target
        self.gamma = gamma
        self.alpha_t = alpha_target

    def quantile(self, calib_scores) -> float:
        return split_conformal_quantile(calib_scores, min(max(self.alpha_t, 0.0), 1.0))

    def update(self, covered: bool) -> None:
        err = 0.0 if covered else 1.0
        self.alpha_t += self.gamma * (self.target - err)


def run_aci(scores, *, calib_window: int, alpha: float = 0.1, gamma: float = 0.05) -> dict:
    """Run ACI online over a time-ordered nonconformity-score stream using a trailing calibration
    window. Returns realized coverage, mean (finite) width, the alpha_t path, and per-step covered."""
    s = np.asarray(scores, float)
    n = len(s)
    aci = ACI(alpha, gamma)
    covered, widths, alpha_path = [], [], []
    for t in range(calib_window, n):
        q = aci.quantile(s[t - calib_window:t])
        cov = bool(s[t] <= q)
        covered.append(cov)
        widths.append(q)
        alpha_path.append(aci.alpha_t)
        aci.update(cov)
    cov_arr = np.asarray(covered, float)
    finite = np.asarray(widths, float)[np.isfinite(widths)]
    return {"coverage": float(cov_arr.mean()) if len(cov_arr) else float("nan"),
            "target_coverage": 1.0 - alpha,
            "mean_width": float(finite.mean()) if len(finite) else float("inf"),
            "covered": cov_arr, "alpha_path": np.asarray(alpha_path)}


def rolling_coverage(covered, window: int):
    """Trailing-window empirical coverage (NaN until the window fills) -- the local-coverage report."""
    c = np.asarray(covered, float)
    out = np.full(len(c), np.nan)
    for i in range(window, len(c) + 1):
        out[i - 1] = c[i - window:i].mean()
    return out


def crisis_coverage(covered, crisis_mask) -> float:
    """Empirical coverage restricted to flagged crisis periods (where bands typically fail)."""
    c, m = np.asarray(covered, bool), np.asarray(crisis_mask, bool)
    return float(c[m].mean()) if m.any() else float("nan")


def kupiec_pof(n: int, failures: int, alpha: float) -> dict:
    """Kupiec proportion-of-failures LR test for unconditional coverage. H0: true miss rate = alpha.
    Returns the LR statistic (~chi2_1) and p-value; small p => coverage is off."""
    from scipy import stats
    if n == 0:
        return {"lr": float("nan"), "pvalue": float("nan"), "miss_rate": float("nan")}
    if not (0.0 < alpha < 1.0):                            # log(alpha)/log(1-alpha) undefined otherwise
        return {"lr": float("nan"), "pvalue": float("nan"), "miss_rate": float(failures / n)}
    x, pi = failures, failures / n
    if x == 0 or x == n:                                   # boundary -> use a guarded form
        lr = -2.0 * (n * math.log(1 - alpha) if x == 0 else n * math.log(alpha))
        lr = lr - (-2.0 * 0.0)
    else:
        ll0 = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        ll1 = (n - x) * math.log(1 - pi) + x * math.log(pi)
        lr = -2.0 * (ll0 - ll1)
    return {"lr": float(lr), "pvalue": float(1.0 - stats.chi2.cdf(lr, df=1)), "miss_rate": float(pi)}
