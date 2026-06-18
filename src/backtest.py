"""Backtest-overfitting & honesty harness (platform Layer 3b / P7 Tier 1).

The integrity backbone of the predictive layer: deterministic, peer-reviewed controls that decide
whether a measured edge is *real* or a multiple-testing fluke -- BEFORE any signal is surfaced.
Everything here is deterministic Python (numpy + scipy); no model, no LLM, no fabricated numbers.

Verified evidence (docs/RESEARCH.md C+):
 - Probabilistic / Deflated Sharpe Ratio (Bailey & Lopez de Prado, J. Portfolio Mgmt 40(5) 2014;
   SSRN 2460551): deflate an observed Sharpe for multiple testing (number of trials N) AND for
   non-normal returns (skew/kurtosis). E[max Sharpe] of N skill-less trials grows ~sqrt(2 ln N).
 - Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation
   (Bailey, Borwein, Lopez de Prado & Zhu, J. Computational Finance 2016; SSRN 2326253):
   PBO = P(the in-sample-best configuration ranks below the OOS median). Caveat: CSCV is biased
   (pessimistic when strategy means ~0, optimistic when one dominates) -- read it as a guardrail.
 - Minimum Backtest Length and "report N" (Pseudo-Mathematics & Financial Charlatanism, AMS 2014).
 - Purged & embargoed K-fold CV (Lopez de Prado, Advances in Financial ML 2018) to stop leakage
   from overlapping labels -- plain k-fold / hold-out CANNOT detect overfitting.
 - Multiple-testing significance bar |t| > 3.0 (Harvey, Liu & Zhu, RFS 2016); FWER (Holm) / FDR (BHY).
 - Post-publication decay haircut ~26% OOS / ~58% post-pub (McLean & Pontiff, JF 2016).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

_EULER = 0.5772156649015329   # Euler-Mascheroni constant (for E[max Sharpe])


# --------------------------------------------------------------------------- #
# Sharpe-ratio significance: Probabilistic & Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns, periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a per-period return series (0 if degenerate)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 and len(r) > 1 else 0.0


def _psr(sr: float, n: int, skew: float, kurt: float, sr_ref: float) -> float:
    """Probabilistic Sharpe Ratio core (all Sharpes PER-PERIOD). P(true SR > sr_ref)."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if n < 2 or denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_ref) * math.sqrt(n - 1) / math.sqrt(denom)))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the strategy's TRUE (per-period) Sharpe exceeds `sr_benchmark`,
    correcting for sample length and non-normal returns (Bailey & Lopez de Prado)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0:
        return float("nan")
    sr = r.mean() / sd                                   # per-period Sharpe
    return _psr(sr, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), sr_benchmark)


def expected_max_sharpe(n_trials: int, sr_trials_std: float) -> float:
    """Expected maximum PER-PERIOD Sharpe across N independent skill-less trials whose Sharpes have
    dispersion `sr_trials_std`. ~ sr_std * [(1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e))] (Bailey & LdP)."""
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    z = stats.norm.ppf
    return float(sr_trials_std * ((1 - _EULER) * z(1 - 1.0 / n_trials)
                                  + _EULER * z(1 - 1.0 / (n_trials * math.e))))


def deflated_sharpe_ratio(returns, n_trials: int, sr_trials_std: float) -> float:
    """DSR: PSR deflated by the multiple-testing benchmark E[max Sharpe] of `n_trials` skill-less
    trials. A DSR near 1 => the edge survives selection bias + non-normality; near 0 => likely a fluke.
    `sr_trials_std` is the dispersion of PER-PERIOD Sharpes across the trials you ran."""
    return probabilistic_sharpe_ratio(returns, sr_benchmark=expected_max_sharpe(n_trials, sr_trials_std))


def min_backtest_length(n_trials: int, target_annual_sharpe: float) -> float:
    """Minimum backtest length (years) so the best of N skill-less trials is not expected to reach
    `target_annual_sharpe`: MinBTL ~ 2 ln(N) / SR^2 (Pseudo-Mathematics, AMS 2014). Report N alongside."""
    if n_trials < 2 or target_annual_sharpe <= 0:
        return float("nan")
    return float(2.0 * math.log(n_trials) / (target_annual_sharpe ** 2))


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)
# --------------------------------------------------------------------------- #
def _col_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column (strategy) over the rows of `block`."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)


def pbo_cscv(perf_matrix, n_splits: int = 16, metric=_col_sharpe) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.

    `perf_matrix`: shape (T, N) -- T time observations x N strategy configurations (per-period pnl).
    Splits T rows into `n_splits` (S, even) disjoint blocks; over all C(S, S/2) train/test combos,
    finds the IS-best strategy and records its OOS relative rank. PBO = fraction of combos where the
    IS-best lands below the OOS median (logit < 0). Returns {pbo, n_combos, logits}.
    """
    M = np.asarray(perf_matrix, dtype=float)
    T, N = M.shape
    S = n_splits if n_splits % 2 == 0 else n_splits - 1
    if S < 2 or N < 2 or T < S:
        return {"pbo": float("nan"), "n_combos": 0, "logits": [], "note": "insufficient data for CSCV"}
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for train_sets in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in train_sets])
        te = np.concatenate([blocks[i] for i in range(S) if i not in train_sets])
        is_best = int(np.argmax(metric(M[tr])))
        oos = metric(M[te])
        # relative rank of the IS-best among OOS performances, in (0,1)
        rank = float(stats.rankdata(oos)[is_best])          # 1..N (ties averaged)
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1 - omega)))
    arr = np.asarray(logits)
    return {"pbo": float(np.mean(arr < 0)), "n_combos": len(logits),
            "median_logit": float(np.median(arr)), "logits": logits}


# --------------------------------------------------------------------------- #
# Purged & embargoed K-fold cross-validation (leakage control)
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K-fold CV for series with overlapping labels (Lopez de Prado, AFML ch.7).

    Each sample i spans [i, t1[i]] (t1[i] = the index at which its label is realized). Training
    samples whose label window overlaps the contiguous test fold are PURGED, and an `embargo`
    fraction of samples immediately after the test fold is dropped, to prevent look-ahead leakage.
    """

    def __init__(self, n_splits: int = 5, embargo: float = 0.01):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, t1):
        t1 = np.asarray(t1, dtype=int)
        n = len(t1)
        emb = int(n * self.embargo)
        idx = np.arange(n)
        for test in np.array_split(idx, self.n_splits):
            t0, t1max = test[0], test[-1]
            # purge: drop train samples whose label window [i, t1[i]] overlaps [t0, t1max]
            keep = ~((t1 >= t0) & (idx <= t1max))
            # embargo: also drop the `emb` samples immediately after the test block
            if emb > 0:
                keep[t1max + 1: t1max + 1 + emb] = False
            keep[t0: t1max + 1] = False                      # never train on the test block itself
            yield idx[keep], test


# --------------------------------------------------------------------------- #
# Multiple-testing corrections (FWER: Holm; FDR: BHY) + significance gate
# --------------------------------------------------------------------------- #
def multipletests(pvals, alpha: float = 0.05, method: str = "bhy"):
    """Adjust p-values for multiple testing. method: 'bonferroni' | 'holm' (FWER) | 'bhy' (FDR,
    Benjamini-Hochberg-Yekutieli, valid under dependence). Returns (reject: bool[], p_adj: float[])."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([])
    if method == "bonferroni":
        p_adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        order = np.argsort(p)
        p_sorted = p[order]
        adj = np.maximum.accumulate((m - np.arange(m)) * p_sorted)
        p_adj = np.empty(m)
        p_adj[order] = np.minimum(adj, 1.0)
    elif method == "bhy":
        order = np.argsort(p)
        p_sorted = p[order]
        cm = np.sum(1.0 / np.arange(1, m + 1))               # Yekutieli dependence correction c(m)
        ranks = np.arange(1, m + 1)
        adj = p_sorted * m * cm / ranks
        adj = np.minimum.accumulate(adj[::-1])[::-1]          # enforce monotonicity from the top
        p_adj = np.empty(m)
        p_adj[order] = np.minimum(adj, 1.0)
    else:
        raise ValueError(f"unknown method {method!r}")
    return p_adj <= alpha, p_adj


def significance_gate(t_stats, threshold: float = 3.0) -> dict:
    """The |t| > 3.0 acceptance bar for a newly discovered signal (Harvey, Liu & Zhu, RFS 2016).
    Returns per-signal pass flags + a two-sided p-value (normal approx) for downstream FDR control."""
    t = np.asarray(t_stats, dtype=float)
    passed = np.abs(t) > threshold
    pvals = 2.0 * stats.norm.sf(np.abs(t))
    return {"threshold": threshold, "passed": passed.tolist(),
            "pvals": pvals.tolist(), "n_passed": int(passed.sum()), "n": int(len(t))}


def decay_haircut(edge: float, *, published: bool = False, illiquid: bool = False,
                  oos_decay: float = 0.26, postpub_decay: float = 0.58) -> dict:
    """Haircut a backtested edge for expected real-world decay (McLean & Pontiff, JF 2016):
    ~26% out-of-sample (overfitting/statistical bias), ~58% once a predictor is public. Illiquid /
    small-cap / high-vol signals decay harder, so an extra penalty is applied. Returns the haircut
    edge plus the multiplier used -- surface the DECAYED estimate, never the raw backtest figure."""
    decay = postpub_decay if published else oos_decay
    if illiquid:
        decay = min(1.0, decay + 0.15)
    mult = max(0.0, 1.0 - decay)
    return {"raw_edge": edge, "haircut_edge": edge * mult, "decay": decay, "multiplier": mult}
