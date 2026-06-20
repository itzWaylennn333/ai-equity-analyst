"""Unit checks for P7 Tier 3 conformal prediction. Deterministic (numpy + scipy)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import conformal as cf  # noqa: E402


def test_split_conformal_marginal_coverage():
    rng = np.random.default_rng(0)
    calib = np.abs(rng.standard_normal(1000))
    q = cf.split_conformal_quantile(calib, alpha=0.1)
    test = np.abs(rng.standard_normal(5000))
    coverage = float(np.mean(test <= q))
    assert 0.86 <= coverage <= 0.94, coverage          # ~90% marginal coverage under exchangeability


def test_quantile_edge_cases():
    assert cf.split_conformal_quantile([], 0.1) == float("inf")
    assert cf.split_conformal_quantile([0.5] * 5, 0.0) == float("inf")   # alpha 0 -> cover all
    assert cf.split_conformal_quantile([0.5] * 5, 1.0) == float("-inf")  # alpha 1 -> empty


def test_weighted_quantile_and_kupiec_edges():
    assert cf.weighted_conformal_quantile([0.1, 0.2, 0.3], alpha=1.0) == float("-inf")   # empty
    assert cf.weighted_conformal_quantile([0.1, 0.2, 0.3], alpha=0.0) == float("inf")    # cover all
    import math
    assert math.isnan(cf.kupiec_pof(n=100, failures=10, alpha=0.0)["lr"])                # guarded, no crash
    assert math.isnan(cf.kupiec_pof(n=100, failures=10, alpha=1.0)["lr"])


def test_interval_width_scales_with_vol():
    lo1, hi1 = cf.conformal_interval(0.0, q=1.0, sigma=1.0)
    lo2, hi2 = cf.conformal_interval(0.0, q=1.0, sigma=3.0)
    assert (hi2 - lo2) > (hi1 - lo1)                    # vol-normalized score -> wider band where vol higher


def test_aci_recovers_coverage_under_shift():
    rng = np.random.default_rng(1)
    scores = np.concatenate([np.abs(rng.standard_normal(1000)),       # calm regime
                             np.abs(rng.standard_normal(1000) * 3)])  # volatility jump
    # A FIXED early quantile badly under-covers the shifted second half:
    q_naive = cf.split_conformal_quantile(scores[:250], alpha=0.1)
    naive_cov2 = float(np.mean(scores[1000:] <= q_naive))
    assert naive_cov2 < 0.7, naive_cov2
    # ACI (rolling window + online alpha update) restores ~target long-run coverage:
    out = cf.run_aci(scores, calib_window=250, alpha=0.1, gamma=0.05)
    assert 0.85 <= out["coverage"] <= 0.95, out["coverage"]


def test_rolling_and_crisis_coverage():
    covered = np.array([True] * 80 + [False] * 20)
    roll = cf.rolling_coverage(covered, window=50)
    assert np.isnan(roll[0]) and abs(roll[49] - 1.0) < 1e-9 and roll[-1] < 1.0
    mask = np.array([False] * 80 + [True] * 20)
    assert cf.crisis_coverage(covered, mask) == 0.0     # all misses fall in the crisis window


def test_kupiec_pof():
    good = cf.kupiec_pof(n=1000, failures=100, alpha=0.1)   # miss rate exactly target
    bad = cf.kupiec_pof(n=1000, failures=200, alpha=0.1)    # double the target miss rate
    assert good["pvalue"] > 0.5 and bad["pvalue"] < 0.05


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
