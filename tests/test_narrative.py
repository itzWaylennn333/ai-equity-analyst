"""Integration check for the narrative drafting agent (P6). Requires local Ollama (bge-m3 + LLM).

Run:  python tests/test_narrative.py   (SKIPS cleanly if Ollama is unreachable)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rag  # noqa: E402
import narrative  # noqa: E402
from test_rag import CFG, TICKER, CHUNKS, _ollama_up  # noqa: E402


def test_draft_section_is_grounded_and_cited():
    rag.build_index(CFG, TICKER, CHUNKS)
    sec = narrative.draft_section(CFG, TICKER, "liquidity and cash position", chunks=CHUNKS)
    assert sec["draft"], sec                                   # non-empty prose
    assert sec["sources"], sec                                 # at least one citation source
    valid_ids = {c["chunk_id"] for c in CHUNKS}
    assert all(s["chunk_id"] in valid_ids for s in sec["sentences"]), sec["sentences"]
    assert all(s["chunk_id"] in valid_ids for s in sec["sources"]), sec["sources"]
    assert 0.0 <= sec["grounded_ratio"] <= 1.0


def test_draft_abstains_off_topic():
    rag.build_index(CFG, TICKER, CHUNKS)
    sec = narrative.draft_section(CFG, TICKER, "the chemical composition of seawater", chunks=CHUNKS)
    assert sec["draft"] == "" and sec.get("note"), sec        # low confidence -> abstain, empty draft


if __name__ == "__main__":
    if not _ollama_up():
        print("SKIP: Ollama not reachable on :11434 (narrative test needs bge-m3 + the LLM)")
        sys.exit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {repr(e)[:300]}")
    shutil.rmtree("/tmp/rag_test_index", ignore_errors=True)
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
