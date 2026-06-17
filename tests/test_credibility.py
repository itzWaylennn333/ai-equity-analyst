"""Unit checks for the credibility & noise layer (platform Layer 2b).

Run:  python tests/test_credibility.py      (prints PASS/FAIL, exits non-zero on failure)
Also importable by pytest (test_* functions with asserts).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import credibility as cred  # noqa: E402

CFG = {"credibility": {"enabled": True, "min_relevance": 0.0, "dedup_jaccard": 0.85,
                       "min_alpha_ratio": 0.55, "min_words": 25}}

_PROSE_BULL = ("PayPal reported strong revenue growth this quarter with expanding operating "
               "margins and robust Braintree volume; management raised full-year guidance and "
               "highlighted improving transaction take rate across its core checkout segment.")
# A true copy-paste near-duplicate of _PROSE_BULL (same text + a short tail), the only kind the
# conservative dedup should suppress -- paraphrases are intentionally NOT flagged (see ARCHITECTURE s5).
_PROSE_BULL_DUP = _PROSE_BULL + " Bullish."
_PROSE_BEAR = ("PayPal faces intensifying competition and pricing pressure in branded checkout; "
               "unbranded take rates are declining, litigation risk persists, and management warned "
               "of softer transaction margins and slowing active-account growth into next year.")
_XBRL_SOUP = ("pypl-20251231 0001633917 2025 FY false 411 481 382 P1D P3D P5D P1Y "
              "http://fasb.org/us-gaap/2025 0.001 0.0001 1000000 utr:Y iso4217:USD xbrli:shares")
_SHORT = "Revenue up."


def _chunks():
    mk = lambda cid, txt, dt="filing": {"chunk_id": cid, "ticker": "PYPL", "doc_type": dt,
                                        "source_file": "10-K.html", "section": "body",
                                        "n_words": len(txt.split()), "text": txt}
    return [mk("A", _PROSE_BULL), mk("B", _PROSE_BULL_DUP), mk("C", _XBRL_SOUP),
            mk("D", _PROSE_BEAR), mk("E", _SHORT)]


def _scores():  # aligns with _chunks(); shape mirrors sentiment.score_text_*
    return [{"tone": 0.6, "polar": 1}, {"tone": 0.6, "polar": 1}, {"tone": 0.0, "polar": 0},
            {"tone": -0.4, "polar": 1}, {"tone": 0.1, "polar": 1}]


def test_helpers():
    assert cred.detect_language(_PROSE_BULL) == "en"
    assert cred.relevance(_PROSE_BULL, "PYPL", aliases=("paypal",)) > 0.5
    assert cred._alpha_ratio(_XBRL_SOUP) < 0.55          # number/tag soup -> low-info
    assert cred._alpha_ratio(_PROSE_BULL) > 0.7


def test_quality_gate():
    gate = {g["chunk_id"]: g for g in cred.quality_gate(_chunks(), CFG, "PYPL", aliases=("paypal",))}
    assert gate["B"]["duplicate_of"] == "A" and "duplicate" in gate["B"]["reasons"]
    assert "low-info" in gate["C"]["reasons"] and not gate["C"]["keep"]
    assert "too-short" in gate["E"]["reasons"] and not gate["E"]["keep"]
    assert gate["A"]["keep"] and gate["D"]["keep"]
    assert gate["A"]["quality"] > 0.0 and gate["C"]["quality"] == 0.0


def test_assess_aggregate():
    rep = cred.assess(CFG, "PYPL", _chunks(), _scores(), aliases=("paypal",))
    # Kept = A (+0.6) and D (-0.4); B dropped (dup), C/E dropped. Weighted tone is between them.
    assert rep["quality"]["kept"] == 2 and rep["quality"]["dropped"] == 3
    assert -0.4 <= rep["weighted_tone"] <= 0.6
    assert rep["quality"]["duplicate_rate"] == 0.2 and rep["quality"]["low_info_rate"] == 0.2
    assert {e["chunk_id"] for e in rep["excluded"]} == {"B", "C", "E"}
    assert rep["manipulation_risk"]["level"] in ("low", "elevated", "high")
    assert rep["social"]["available"] is False
    assert rep["top_weighted"][0]["chunk_id"] == "A"   # highest credibility weight


def test_social_scaffold_raises():
    for fn in (cred.account_credibility, cred.coordination):
        try:
            fn([{"account": "x"}], CFG)
            raise AssertionError(f"{fn.__name__} should raise NotImplementedError")
        except NotImplementedError:
            pass


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
