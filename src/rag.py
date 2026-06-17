"""Retrieval-augmented extraction (platform Layer 3 / P6) -- grounded, cited, local-only.

Hybrid retrieval over ingested document chunks, then schema-constrained extraction that MUST
cite a verbatim source span -- "not found" over a guess (integrity rules). Runs entirely on
the Spark: dense embeddings via Ollama (bge-m3), vector store in embedded LanceDB, sparse BM25
in pure Python, fused with Reciprocal Rank Fusion. No torch; no network beyond local Ollama.

Evidence base (docs/RESEARCH.md): retrieval -- not generation -- is the dominant failure mode
in financial-filing QA (FinanceBench arXiv:2311.11944; FinDER arXiv:2504.15800), and
chunk-level grounding is the hard part (FinAgentBench arXiv:2508.14052). So we (1) hybridise
dense + sparse, (2) gate generation on retrieval confidence, and (3) verify every extracted
figure appears verbatim in the cited chunk before returning it.

Roadmap (not yet here): cross-encoder reranking (bge-reranker-v2-m3), RAPTOR tree index,
Self-RAG-style self-critique + citation precision/recall. See docs/ARCHITECTURE.md s11.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import requests

import utils

try:
    import ingest
except Exception:  # pragma: no cover
    ingest = None

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-%$]*")

# Schema-constrained extraction output (Ollama `format`); mirrors the grounded-citation contract.
_EXTRACT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "found": {"type": "boolean"},
        "value": {"type": "string"},     # figure copied VERBATIM as it appears in the passage
        "unit": {"type": "string"},
        "period": {"type": "string"},
        "passage": {"type": "integer"},  # 1-based index of the supporting passage
        "quote": {"type": "string"},     # exact sentence/span copied from that passage
        "rationale": {"type": "string"},
    },
    "required": ["found", "value", "passage", "quote", "rationale"],
}


def _cfg(cfg: dict) -> dict:
    c = dict(cfg.get("rag") or {})
    s = cfg.get("sentiment") or {}
    fb = s.get("llm_endpoint") or "http://localhost:11434"
    # `or`-coalesce the string knobs so an explicit YAML `null` falls back (setdefault would keep None).
    c["embed_model"] = c.get("embed_model") or "bge-m3"
    c["embed_endpoint"] = c.get("embed_endpoint") or fb
    c["llm_endpoint"] = c.get("llm_endpoint") or fb
    c["llm_model"] = c.get("llm_model") or s.get("llm_model") or "qwen3:30b-a3b-instruct-2507-q4_K_M"
    c["index_dir"] = c.get("index_dir") or "data/processed/{ticker}/lancedb"
    c.setdefault("top_k", 5)
    c.setdefault("candidates", 30)        # per-retriever shortlist before fusion
    c.setdefault("rrf_k", 60)             # RRF constant (Cormack et al. 2009)
    c.setdefault("min_similarity", 0.45)  # retrieval-confidence gate (top cosine); below -> "not found"
    c["rerank"] = c.get("rerank") or "llm"  # "llm" (torch-free listwise) | "none" | "cross-encoder" (future)
    c.setdefault("rerank_pool", 12)       # fused candidates fed to the reranker, narrowed to top_k
    c.setdefault("verify_support", True)  # LLM-as-judge faithfulness check on the extracted figure
    return c


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def _index_path(cfg: dict, ticker: str):
    return utils.resolve(_cfg(cfg)["index_dir"].format(ticker=ticker))


# --------------------------------------------------------------------------- #
# Embeddings (Ollama bge-m3) + index (LanceDB)
# --------------------------------------------------------------------------- #
def embed(texts: list[str], cfg: dict, *, batch: int = 16) -> list[list[float]]:
    c = _cfg(cfg)
    url = f"{c['embed_endpoint'].rstrip('/')}/api/embed"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        payload = {"model": c["embed_model"], "input": [(t or " ")[:8000] for t in texts[i:i + batch]]}
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out


def build_index(cfg: dict, ticker: str, chunks: list[dict]) -> dict:
    """Embed chunks and (re)build the per-ticker LanceDB vector table."""
    import lancedb
    path = _index_path(cfg, ticker)
    path.mkdir(parents=True, exist_ok=True)
    vecs = embed([ch.get("text", "") for ch in chunks], cfg)
    rows = [{"chunk_id": ch.get("chunk_id"), "text": ch.get("text", ""),
             "doc_type": ch.get("doc_type", "other"), "section": ch.get("section", ""),
             "source_file": ch.get("source_file", ""), "vector": v}
            for ch, v in zip(chunks, vecs)]
    db = lancedb.connect(str(path))
    db.create_table("chunks", data=rows, mode="overwrite")
    return {"ticker": ticker, "n_chunks": len(rows), "dim": len(vecs[0]) if vecs else 0,
            "path": str(path)}


def _rrf(rankings: list[list[str]], k: int) -> dict:
    """Reciprocal Rank Fusion over best-first id rankings. score(id)=sum 1/(k+rank)."""
    score: dict = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] += 1.0 / (k + rank + 1)
    return score


def _chat_json(cfg: dict, prompt: str, schema: dict, *, timeout: int = 120) -> dict:
    """Schema-constrained Ollama chat -> parsed dict (raises on call/parse failure)."""
    c = _cfg(cfg)
    body = {"model": c["llm_model"], "stream": False, "format": schema,
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(f"{c['llm_endpoint'].rstrip('/')}/api/chat", timeout=timeout, json=body)
    r.raise_for_status()
    msg = r.json().get("message", {}) or {}
    return json.loads(msg.get("content") or msg.get("thinking") or "")


_RERANK_SCHEMA = {"type": "object", "additionalProperties": False,
                  "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
                  "required": ["ranking"]}


def _llm_rerank(cfg: dict, query: str, passages: list[dict], k: int) -> list[dict]:
    """Listwise LLM reranker — a torch-free stand-in for a cross-encoder (bge-reranker-v2-m3).
    Reorders passages best-first and truncates to k; falls back to input order on any failure."""
    listing = "\n\n".join(f"[{i + 1}] {(p.get('text') or '')[:600]}" for i, p in enumerate(passages))
    prompt = ("Rank the passages by how directly they help answer the QUERY. Return JSON "
              f'{{"ranking": [passage numbers, most relevant first]}} with the top {k}; use only '
              "the numbers shown.\n\nQUERY: " + query + "\n\nPASSAGES:\n" + listing)
    try:
        order = [i - 1 for i in _chat_json(cfg, prompt, _RERANK_SCHEMA).get("ranking", [])
                 if isinstance(i, int) and 1 <= i <= len(passages)]
        seen: set = set()
        order = [i for i in order if not (i in seen or seen.add(i))]   # dedupe, preserve order
        if not order:
            return passages[:k]
        reranked = [passages[i] for i in order] + [p for j, p in enumerate(passages) if j not in seen]
        return reranked[:k]
    except Exception:
        return passages[:k]


_SUPPORT_SCHEMA = {"type": "object", "additionalProperties": False,
                   "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
                   "required": ["supported", "reason"]}


def _judge_support(cfg: dict, question: str, value: str, quote: str):
    """Self-RAG-style faithfulness check: does the QUOTE actually support VALUE as the answer to
    QUESTION (not just coincidentally contain the number)? Catches 'right chunk, wrong figure'."""
    prompt = ("Decide whether the QUOTE directly supports the ANSWER to the QUESTION — i.e. the quote "
              "states that figure as the answer, not merely contains the number coincidentally. "
              'Return JSON {"supported": bool, "reason": str}.\n\n'
              f"QUESTION: {question}\nANSWER: {value}\nQUOTE: {quote}")
    try:
        d = _chat_json(cfg, prompt, _SUPPORT_SCHEMA)
        return bool(d.get("supported")), d.get("reason", "")
    except Exception as e:
        return None, f"support-judge unavailable: {e}"   # None = couldn't verify (don't hard-fail)


def citation_metrics(gold_chunk_ids, cited_chunk_ids) -> dict:
    """Citation precision/recall/F1 vs a gold set (ALCE-style), for evaluating grounded extraction."""
    gold, cited = set(gold_chunk_ids or []), set(cited_chunk_ids or [])
    if not cited:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(gold & cited)
    p = tp / len(cited)
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


# --------------------------------------------------------------------------- #
# Hybrid retrieval (dense + BM25, RRF-fused) -> rerank -> confidence gate
# --------------------------------------------------------------------------- #
def retrieve(cfg: dict, ticker: str, query: str, *, k: int | None = None,
             chunks: list[dict] | None = None) -> dict:
    import lancedb
    from rank_bm25 import BM25Okapi
    c = _cfg(cfg)
    k = k or c["top_k"]
    n = c["candidates"]
    if chunks is None:
        chunks = ingest.load_chunks(cfg, ticker) if ingest else []
    by_id = {ch["chunk_id"]: ch for ch in chunks}

    # Dense: query embedding -> LanceDB cosine search.
    qv = embed([query], cfg)[0]
    tbl = lancedb.connect(str(_index_path(cfg, ticker))).open_table("chunks")
    dense = tbl.search(qv).metric("cosine").limit(n).to_list()
    dense_ids = [r["chunk_id"] for r in dense]
    top_sim = (1.0 - dense[0]["_distance"]) if dense else 0.0

    # Sparse: BM25 over the same corpus.
    bm = BM25Okapi([_tokenize(ch.get("text", "")) for ch in chunks])
    bm_scores = bm.get_scores(_tokenize(query))
    sparse_ids = [chunks[i]["chunk_id"] for i in sorted(range(len(chunks)), key=lambda i: -bm_scores[i])[:n]]

    fused = _rrf([dense_ids, sparse_ids], c["rrf_k"])
    pool_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:c["rerank_pool"]]]
    pool = [{"chunk_id": cid, "rrf": round(fused[cid], 5),
             **{f: by_id.get(cid, {}).get(f) for f in ("text", "doc_type", "section", "source_file")}}
            for cid in pool_ids]
    do_rerank = c["rerank"] == "llm" and len(pool) > 1
    results = _llm_rerank(cfg, query, pool, k) if do_rerank else pool[:k]
    return {"query": query, "top_similarity": round(float(top_sim), 3),
            "confident": float(top_sim) >= c["min_similarity"], "n_candidates": len(chunks),
            "reranked": bool(do_rerank), "results": results}


# --------------------------------------------------------------------------- #
# Grounded extraction: cite-or-abstain, with verbatim span verification
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def extract(cfg: dict, ticker: str, question: str, *, k: int | None = None,
            chunks: list[dict] | None = None) -> dict:
    """Retrieve, then extract a single figure that MUST be quoted from a cited chunk.

    Returns found=False (never a fabricated/inferred number) when retrieval confidence is low,
    the model abstains, or the quoted value/sentence cannot be verified verbatim in the cited chunk.
    """
    c = _cfg(cfg)
    ret = retrieve(cfg, ticker, question, k=k, chunks=chunks)
    if not ret["results"] or not ret["confident"]:
        return {"found": False, "answer": None, "citation": None, "retrieval": ret,
                "reason": f"insufficient retrieval confidence (top cosine {ret['top_similarity']} "
                          f"< {c['min_similarity']})"}
    passages = ret["results"]
    listing = "\n\n".join(
        f"[{i + 1}] (chunk_id={p['chunk_id']}, {p['source_file']} / {p['section']})\n{(p['text'] or '')[:1500]}"
        for i, p in enumerate(passages))
    prompt = ("You are an equity analyst. Using ONLY the numbered passages below, answer the question "
              "with a single figure. Copy the supporting figure VERBATIM as it appears in the text "
              "(same digits, units, and formatting). Cite the passage number you used and quote the exact "
              "sentence containing the figure. If the passages do not contain the answer, set found=false. "
              "Never compute, infer, or estimate a number that is not literally present.\n\n"
              f"QUESTION: {question}\n\nPASSAGES:\n{listing}")
    body = {"model": c["llm_model"], "stream": False, "format": _EXTRACT_SCHEMA,
            "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post(f"{c['llm_endpoint'].rstrip('/')}/api/chat", timeout=120, json=body)
        r.raise_for_status()
        msg = r.json().get("message", {}) or {}
        data = json.loads(msg.get("content") or msg.get("thinking") or "")
    except Exception as e:
        return {"found": False, "answer": None, "citation": None, "retrieval": ret,
                "reason": f"extraction call/parse failed: {e}"}

    if not data.get("found"):
        return {"found": False, "answer": None, "citation": None, "retrieval": ret,
                "reason": data.get("rationale") or "model found no answer in the passages"}
    pi = data.get("passage", 0)
    if not isinstance(pi, int) or not (1 <= pi <= len(passages)):
        return {"found": False, "answer": None, "citation": None, "retrieval": ret,
                "reason": f"invalid passage citation ({pi})"}
    src = passages[pi - 1]
    quote, value = data.get("quote") or "", data.get("value") or ""
    quote_ok = bool(quote) and _norm(quote) in _norm(src["text"])
    value_ok = bool(value) and _norm(value) in _norm(src["text"])
    verbatim = quote_ok and value_ok
    # Self-RAG-style faithfulness gate: only run when the span is verbatim-verified.
    supported, support_reason = (None, None)
    if verbatim and c["verify_support"]:
        supported, support_reason = _judge_support(cfg, question, value, quote)
    grounded = verbatim and (supported is not False)   # fail only on EXPLICIT non-support
    if grounded:
        reason = None
    elif verbatim and supported is False:
        reason = "failed support check: " + (support_reason or "quote does not support the answer")
    else:
        reason = "failed verbatim verification (quoted value/sentence not in the cited chunk)"
    return {
        "found": grounded,
        "answer": {"value": value, "unit": data.get("unit"), "period": data.get("period")} if grounded else None,
        "citation": {"chunk_id": src["chunk_id"], "source_file": src["source_file"],
                     "section": src["section"], "quote": quote} if grounded else None,
        "verification": {"quote_in_source": quote_ok, "value_in_source": value_ok,
                         "supported": supported, "support_reason": support_reason},
        "rationale": data.get("rationale"),
        "reason": reason,
        "retrieval": ret,
    }
