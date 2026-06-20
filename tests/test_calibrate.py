"""Unit checks for P7 Tier 2 probability calibration. Deterministic (numpy + sklearn)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import calibrate as cal  # noqa: E402


def test_brier_and_ece_basics():
    assert cal.brier_score([1, 0], [1, 0]) == 0.0
    assert abs(cal.brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-12
    # confidently wrong -> large ECE; perfectly matched bins -> ~0
    assert cal.ece(np.full(200, 0.9), np.zeros(200)) > 0.8


def test_choose_method_sample_size_rule():
    assert cal.choose_method(500) == "platt"
    assert cal.choose_method(1000) == "isotonic"
    assert cal.choose_method(5000) == "isotonic"


def test_calibration_reduces_ece():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(int)
    raw = true_p ** 2                                  # monotonic but miscalibrated (systematically low)
    cs, ts = raw[:2000], raw[2000:]
    cy, ty = y[:2000], y[2000:]
    ece_raw = cal.ece(ts, ty)
    for method in ("platt", "isotonic", "beta"):
        c = cal.fit_calibrator(cs, cy, method=method)
        assert cal.ece(c.predict(ts), ty) < ece_raw, method


def test_beta_never_worse_on_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)        # raw scores already = true prob
    cs, ts, cy, ty = p[:2000], p[2000:], y[:2000], y[2000:]
    beta = cal.fit_calibrator(cs, cy, method="beta")
    assert cal.ece(beta.predict(ts), ty) <= cal.ece(ts, ty) + 0.03   # contains identity -> not worse


def test_rolling_metric_flags_miscalibration():
    r = cal.rolling_metric(np.full(300, 0.9), np.zeros(300), window=100, metric="ece")
    assert np.nanmax(r) > 0.5


def test_calibration_report_structure():
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 1, 600)
    y = (rng.uniform(size=600) < s).astype(int)
    rep = cal.calibration_report(s, y, train_window=200, monitor_window=150)
    assert rep["n_oos"] == 400 and len(rep["probs"]) == 400
    assert 0.0 <= rep["oos_ece"] <= 1.0 and isinstance(rep["drift"], bool)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {repr(e)[:300]}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
