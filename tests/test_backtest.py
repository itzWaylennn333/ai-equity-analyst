"""Unit checks for the P7 backtest-overfitting / honesty harness. Pure & deterministic (numpy/scipy).

Run:  python tests/test_backtest.py    (also pytest-compatible)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import backtest as bt  # noqa: E402


def test_sharpe_ratio():
    assert bt.sharpe_ratio(np.zeros(252)) == 0.0  # degenerate (zero mean & std) -> 0 by guard
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 2520)
    assert bt.sharpe_ratio(x) > 0                 # positive-mean series -> positive Sharpe


def test_psr_and_dsr_monotonic_in_trials():
    rng = np.random.default_rng(1)
    good = rng.normal(0.001, 0.01, 2000)          # per-period Sharpe ~0.1
    assert bt.probabilistic_sharpe_ratio(good, sr_benchmark=0.0) > 0.9   # clearly > 0
    dsr_few = bt.deflated_sharpe_ratio(good, n_trials=2, sr_trials_std=0.05)
    dsr_many = bt.deflated_sharpe_ratio(good, n_trials=1000, sr_trials_std=0.05)
    assert dsr_few > dsr_many                      # more trials tested -> harder to clear (deflation)
    assert dsr_few > 0.5 and dsr_many < dsr_few


def test_expected_max_and_minbtl():
    assert bt.expected_max_sharpe(1000, 0.05) > bt.expected_max_sharpe(10, 0.05) > 0
    assert bt.min_backtest_length(1000, 1.0) > bt.min_backtest_length(10, 1.0)      # more trials -> longer
    assert bt.min_backtest_length(100, 0.5) > bt.min_backtest_length(100, 2.0)      # weaker target -> longer


def test_pbo_separates_skill_from_noise():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((240, 20))                       # all zero-mean -> overfit prone
    pbo_noise = bt.pbo_cscv(noise, n_splits=8)["pbo"]
    skill = rng.standard_normal((240, 20))
    skill[:, 0] += 0.5                                           # strategy 0 has a real edge
    pbo_skill = bt.pbo_cscv(skill, n_splits=8)["pbo"]
    assert pbo_skill < 0.1                                       # genuine edge -> low PBO
    assert pbo_noise > 0.2                                       # noise -> materially elevated PBO
    assert pbo_noise - pbo_skill > 0.15                          # the harness SEPARATES skill from noise


def test_purged_kfold_no_leakage():
    n = 100
    t1 = np.arange(n) + 5                                         # each label spans 5 steps ahead
    for train, test in bt.PurgedKFold(n_splits=5, embargo=0.05).split(t1):
        assert set(train).isdisjoint(set(test))                  # never train on the test block
        t0, tmax = test[0], test[-1]
        for i in train:                                          # no label window overlaps the test fold
            assert (t1[i] < t0) or (i > tmax), (i, t1[i], t0, tmax)


def test_multipletests():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    rej_b, adj_b = bt.multipletests(p, alpha=0.05, method="bonferroni")
    assert np.allclose(adj_b, [0.05, 0.10, 0.15, 0.20, 0.25])
    assert rej_b.tolist() == [True, False, False, False, False]
    rej_h, adj_h = bt.multipletests(p, alpha=0.05, method="holm")
    assert rej_h.tolist() == [True, False, False, False, False]
    rej_y, adj_y = bt.multipletests(p, alpha=0.05, method="bhy")
    assert ((adj_y >= 0) & (adj_y <= 1)).all() and len(adj_y) == 5   # BHY is more conservative


def test_significance_gate():
    g = bt.significance_gate([3.5, 2.0, -3.1, 1.0], threshold=3.0)
    assert g["passed"] == [True, False, True, False] and g["n_passed"] == 2


def test_decay_haircut():
    assert abs(bt.decay_haircut(1.0)["multiplier"] - 0.74) < 1e-9            # OOS decay 26%
    assert abs(bt.decay_haircut(1.0, published=True)["multiplier"] - 0.42) < 1e-9   # post-pub 58%
    assert bt.decay_haircut(1.0, published=True, illiquid=True)["multiplier"] < 0.42  # extra penalty


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
