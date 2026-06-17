"""Unit checks for the reconciliation agent (P6.3). Pure logic -- no Ollama needed.

Run:  python tests/test_reconcile.py    (also pytest-compatible)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reconcile  # noqa: E402


def test_parse_number():
    assert reconcile._parse_number("33,172") == 33172
    assert abs(reconcile._parse_number("$33.2 billion") - 33.2e9) < 1   # float-scale, approx
    assert reconcile._parse_number("(1,234)") == -1234
    assert reconcile._parse_number("$1.5M") == 1.5e6
    assert reconcile._parse_number("approximately 23,800 people globally") == 23800
    assert reconcile._parse_number("7%") == 7
    assert reconcile._parse_number("no number here") is None
    assert reconcile._parse_number("33,172 (basic)") == 33172   # trailing '(basic)' is not a suffix


def test_same_magnitude():
    assert reconcile._same_magnitude(33172, 33172e6, 0.02) == 1e6   # millions vs absolute
    assert reconcile._same_magnitude(33.172, 33172e6, 0.02) == 1e9   # billions vs absolute
    assert reconcile._same_magnitude(33172, 99999e6, 0.02) is None   # genuinely different


def _fake_extractor(mapping):
    def f(cfg, ticker, question):
        for key, res in mapping.items():
            if key in question:
                return res
        return {"found": False, "reason": "no match"}
    return f


def test_reconcile_match_mismatch_notfound():
    items = [
        {"metric": "net revenues", "question": "total net revenues fiscal 2025", "expected": 33172e6},
        {"metric": "net income", "question": "net income fiscal 2025", "expected": 4000e6},
        {"metric": "operating income", "question": "operating income fiscal 2025", "expected": 3000e6},
    ]
    fake = _fake_extractor({
        "net revenues": {"found": True, "answer": {"value": "33,172"}, "citation": {"chunk_id": "c1"}},
        "net income": {"found": True, "answer": {"value": "33,172"}, "citation": {"chunk_id": "c2"}},  # wrong
        "operating income": {"found": False, "reason": "not in passages"},
    })
    out = reconcile.reconcile({}, "TEST", items, extractor=fake)
    by = {r["metric"]: r for r in out["items"]}
    assert by["net revenues"]["status"] == "match" and by["net revenues"]["scale"] == 1e6
    assert by["net income"]["status"] == "mismatch"
    assert by["operating income"]["status"] == "filing_not_found"
    assert out["counts"]["match"] == 1 and out["counts"]["mismatch"] == 1
    assert out["ok"] is False and len(out["conflicts"]) == 1


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
