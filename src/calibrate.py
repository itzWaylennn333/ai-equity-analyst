"""Probability calibration + drift monitoring (platform Layer 3b / P7 Tier 2).

Turns a model's raw scores into *calibrated* probabilities and -- crucially for a regime-prone
domain -- keeps watching whether that calibration still holds out-of-sample. Deterministic
(numpy + scikit-learn); the output is an honest probability with a monitored reliability, never a
point forecast.

Verified evidence (docs/RESEARCH.md C+):
 - Calibrator choice is a SAMPLE-SIZE decision: Platt (sigmoid) wins below ~1000 calibration
   points, isotonic only catches up at >=1000 (Niculescu-Mizil & Caruana, ICML 2005). Equities
   have few independent events -> DEFAULT Platt.
 - Beta calibration is a "never-worse" 3-parameter alternative that contains the identity map, so
   (unlike logistic) it cannot un-calibrate an already-calibrated score (Kull et al., AISTATS 2017).
 - Post-hoc calibration DECAYS under distribution shift; refit on rolling windows and monitor
   ECE/Brier out-of-sample (Ovadia et al., NeurIPS 2019). Never treat a fit-once map as stable.
 - Temperature scaling is for softmax neural nets, not trees -- deliberately omitted (Guo et al. 2017).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Calibration quality metrics
# --------------------------------------------------------------------------- #
def brier_score(probs, outcomes) -> float:
    """Mean squared error of probabilistic predictions (lower = better)."""
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def reliability_curve(probs, outcomes, n_bins: int = 10):
    """Binned reliability: (bin_mean_pred, bin_observed_freq, bin_count) over equal-width bins."""
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred, obs_freq, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        count.append(int(m.sum()))
        mean_pred.append(float(p[m].mean()) if m.any() else float("nan"))
        obs_freq.append(float(y[m].mean()) if m.any() else float("nan"))
    return np.array(mean_pred), np.array(obs_freq), np.array(count)


def ece(probs, outcomes, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean |confidence - accuracy| over bins."""
    p = np.asarray(probs, float)
    mean_pred, obs_freq, count = reliability_curve(p, outcomes, n_bins)
    m = count > 0
    if not m.any():
        return float("nan")
    return float(np.sum(np.abs(mean_pred[m] - obs_freq[m]) * count[m]) / count[m].sum())


# --------------------------------------------------------------------------- #
# Calibrators (each: .fit(scores, y) -> self ; .predict(scores) -> probabilities)
# --------------------------------------------------------------------------- #
class PlattCalibrator:
    """Logistic (Platt/sigmoid) calibration. Default choice for small calibration sets."""

    def fit(self, scores, y):
        s = np.asarray(scores, float).reshape(-1, 1)
        self.lr_ = LogisticRegression(solver="lbfgs").fit(s, np.asarray(y, int))
        return self

    def predict(self, scores):
        return self.lr_.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic-regression calibration. Use only with >=~1000 independent calibration points."""

    def fit(self, scores, y):
        self.ir_ = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(scores, float),
                                                                 np.asarray(y, float))
        return self

    def predict(self, scores):
        return self.ir_.predict(np.asarray(scores, float))


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic regression on [ln s, -ln(1-s)]. Contains the
    identity map, so it is 'never worse' than the raw scores; a safe drop-in for binary signals."""

    def fit(self, scores, y):
        s = np.clip(np.asarray(scores, float), _EPS, 1 - _EPS)
        X = np.column_stack([np.log(s), -np.log(1 - s)])
        self.lr_ = LogisticRegression(solver="lbfgs").fit(X, np.asarray(y, int))
        return self

    def predict(self, scores):
        s = np.clip(np.asarray(scores, float), _EPS, 1 - _EPS)
        X = np.column_stack([np.log(s), -np.log(1 - s)])
        return self.lr_.predict_proba(X)[:, 1]


_CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator, "beta": BetaCalibrator}


def choose_method(n_calibration: int) -> str:
    """Sample-size rule (Niculescu-Mizil & Caruana 2005): isotonic >=1000 pts, else Platt."""
    return "isotonic" if n_calibration >= 1000 else "platt"


def fit_calibrator(scores, y, method: str = "auto"):
    if method == "auto":
        method = choose_method(len(np.asarray(scores)))
    return _CALIBRATORS[method]().fit(scores, y)


# --------------------------------------------------------------------------- #
# Walk-forward calibration + out-of-sample drift monitoring
# --------------------------------------------------------------------------- #
def walk_forward_calibrate(scores, y, *, train_window: int, method: str = "auto") -> dict:
    """Time-ordered walk-forward: fit the calibrator on a trailing `train_window` and predict the
    NEXT point, never shuffling (no leakage). Returns the aligned OOS calibrated probabilities."""
    s, yy = np.asarray(scores, float), np.asarray(y, int)
    n = len(s)
    if n <= train_window:
        return {"start": n, "probs": np.array([]), "y": np.array([]), "note": "series shorter than window"}
    out = np.empty(n - train_window)
    for i in range(train_window, n):
        cal = fit_calibrator(s[i - train_window:i], yy[i - train_window:i], method=method)
        out[i - train_window] = float(cal.predict(s[i:i + 1])[0])
    return {"start": train_window, "probs": out, "y": yy[train_window:]}


def rolling_metric(probs, y, window: int, metric: str = "ece", n_bins: int = 10):
    """Trailing-window calibration metric series ('ece' or 'brier'); NaN until the window fills."""
    p, yy = np.asarray(probs, float), np.asarray(y, float)
    fn = (lambda a, b: ece(a, b, n_bins)) if metric == "ece" else brier_score
    out = np.full(len(p), np.nan)
    for i in range(window, len(p) + 1):
        out[i - 1] = fn(p[i - window:i], yy[i - window:i])
    return out


def calibration_report(scores, y, *, train_window: int, monitor_window: int = 250,
                       method: str = "auto", ece_threshold: float = 0.1) -> dict:
    """End-to-end honesty surface: walk-forward calibrate, then monitor OOS ECE/Brier on a rolling
    window and FLAG drift past `ece_threshold` (widen bands / abstain when flagged)."""
    wf = walk_forward_calibrate(scores, y, train_window=train_window, method=method)
    p, yy = wf["probs"], wf["y"]
    if len(p) == 0:
        return {**wf, "oos_ece": float("nan"), "oos_brier": float("nan"), "drift": True}
    roll_ece = rolling_metric(p, yy, min(monitor_window, len(p)), "ece")
    drift_flags = roll_ece > ece_threshold
    return {
        "method": method if method != "auto" else choose_method(train_window),
        "n_oos": int(len(p)),
        "oos_ece": float(ece(p, yy)),
        "oos_brier": float(brier_score(p, yy)),
        "rolling_ece": roll_ece,
        "drift_flag_rate": float(np.nanmean(drift_flags.astype(float))) if np.isfinite(roll_ece).any() else float("nan"),
        "drift": bool(np.nansum(drift_flags) > 0),
        "probs": p, "y": yy,
    }
