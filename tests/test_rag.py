"""Integration checks for the RAG extraction core (P6).

Requires a local Ollama server with bge-m3 + the configured LLM.
Run:  python tests/test_rag.py    (SKIPS cleanly if Ollama is unreachable)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests  # noqa: E402
import rag  # noqa: E402

CFG = {
    "sentiment": {"llm_endpoint": "http://localhost:11434",
                  "llm_model": "qwen3:30b-a3b-instruct-2507-q4_K_M"},
    "rag": {"embed_model": "bge-m3", "index_dir": "/tmp/rag_test_index/{ticker}",
            "top_k": 4, "min_similarity": 0.4},
}
TICKER = "TEST"
CHUNKS = [
    {"chunk_id": "c1", "ticker": TICKER, "doc_type": "filing", "source_file": "10-K.html", "section": "MD&A",
     "text": "Net revenues for fiscal 2025 were $31.8 billion, an increase of 7% compared to the prior year, "
             "driven by higher total payment volume and an improved transaction take rate."},
    {"chunk_id": "c2", "ticker": TICKER, "doc_type": "filing", "source_file": "10-K.html", "section": "Risk Factors",
     "text": "We face intense competition in the digital payments industry, and our business is subject to "
             "extensive and evolving regulation across multiple jurisdictions, which could adversely affect results."},
    {"chunk_id": "c3", "ticker": TICKER, "doc_type": "filing", "source_file": "10-K.html", "section": "Liquidity",
     "text": "As of December 31, 2025, the company held $12.5 billion in cash, cash equivalents and short-term "
             "investments, and repurchased $6.0 billion of common stock during the year."},
    {"chunk_id": "c4", "ticker": TICKER, "doc_type": "filing", "source_file": "10-K.html", "section": "Business",
     "text": "The company operates a two-sided platform connecting merchants and consumers across more than "
             "200 markets, enabling digital and mobile payments, checkout, and money transfer services."},
]


def _ollama_up() -> bool:
    try:
        requests.get("http://localhost:11434/api/version", timeout=3)
        return True
    except Exception:
        return False


def test_retrieve_ranks_relevant():
    rag.build_index(CFG, TICKER, CHUNKS)
    ret = rag.retrieve(CFG, TICKER, "What were net revenues in fiscal 2025?", chunks=CHUNKS)
    assert ret["confident"], ret
    assert ret["results"][0]["chunk_id"] == "c1", ret["results"]


def test_extract_cites_and_verifies():
    rag.build_index(CFG, TICKER, CHUNKS)
    out = rag.extract(CFG, TICKER, "What were net revenues in fiscal 2025?", chunks=CHUNKS)
    assert out["found"], out
    assert out["citation"]["chunk_id"] == "c1", out
    assert "31.8" in out["answer"]["value"], out
    assert out["verification"]["value_in_source"] and out["verification"]["quote_in_source"], out


def test_abstains_off_topic():
    # Off-topic query -> retrieval confidence below the gate -> must abstain, never fabricate.
    rag.build_index(CFG, TICKER, CHUNKS)
    out = rag.extract(CFG, TICKER, "What is the boiling point of water in Celsius?", chunks=CHUNKS)
    assert out["found"] is False, out


def test_rerank_active_and_keeps_relevant():
    rag.build_index(CFG, TICKER, CHUNKS)
    ret = rag.retrieve(CFG, TICKER, "net revenues for fiscal 2025", chunks=CHUNKS)
    assert ret["reranked"] is True, ret           # default rerank: "llm"
    assert ret["results"][0]["chunk_id"] == "c1", ret["results"]


def test_citation_metrics():  # pure unit (no Ollama needed)
    m = rag.citation_metrics(["c1", "c2"], ["c1", "c3"])
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert rag.citation_metrics([], [])["f1"] == 0.0


if __name__ == "__main__":
    if not _ollama_up():
        print("SKIP: Ollama not reachable on :11434 (RAG tests need bge-m3 + the LLM)")
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
