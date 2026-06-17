"""RAPTOR tree index (platform Layer 3 / P6) -- recursive embed-cluster-summarize. OPTIONAL.

RAPTOR (Sarthi et al., ICLR 2024, arXiv:2401.18059) builds a multi-level tree by recursively
embedding, clustering, and LLM-summarizing chunks, then retrieves across levels of abstraction
(the paper's "collapsed tree" mode). Good for broad, cross-section questions over long filings.

INTEGRITY BOUNDARY (important): summary nodes are LLM-GENERATED, not verbatim source text. They
are for retrieval / navigation ONLY. The cite-or-abstain extractor (rag.extract) must still ground
figures in LEAF chunks -- a summary hit is expanded to its leaf descendants (`leaf_chunk_ids`)
before anything is cited. RAPTOR was also the least finance-benchmarked technique in our research
pass (summary-hallucination + tree-recompute caveats), so it ships OFF by default
(cfg rag.index_mode != "raptor").
"""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

import rag


def _summarize(cfg: dict, texts: list[str]) -> str:
    import requests
    c = rag._cfg(cfg)
    joined = "\n\n---\n\n".join((t or "")[:1500] for t in texts)
    prompt = ("Summarize the key facts, figures and themes from these excerpts of a company's filings "
              "into one faithful, self-contained paragraph. Do not invent or compute numbers not present.\n\n"
              + joined)
    r = requests.post(f"{c['llm_endpoint'].rstrip('/')}/api/chat", timeout=180,
                      json={"model": c["llm_model"], "stream": False,
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return ((r.json().get("message", {}) or {}).get("content") or "").strip()


def _cluster(vecs: list[list[float]], cluster_size: int) -> list[int]:
    """Agglomerative (cosine) clustering into ~n/cluster_size groups. Returns a label per node."""
    n = len(vecs)
    if n <= cluster_size:
        return [0] * n
    from sklearn.cluster import AgglomerativeClustering
    k = max(2, round(n / cluster_size))
    return AgglomerativeClustering(n_clusters=k, metric="cosine",
                                   linkage="average").fit_predict(np.asarray(vecs)).tolist()


def build_tree(cfg: dict, ticker: str, chunks: list[dict], *, max_levels: int = 3,
               cluster_size: int = 6) -> list[dict]:
    """Build the RAPTOR tree; return all nodes (leaves + summaries) as
    {node_id, level, text, is_summary, child_ids, source_file, section}."""
    from collections import defaultdict
    nodes, level_nodes = [], []
    for ch in chunks:
        leaf = {"node_id": ch.get("chunk_id"), "level": 0, "text": ch.get("text", ""),
                "is_summary": False, "child_ids": [], "source_file": ch.get("source_file"),
                "section": ch.get("section")}
        nodes.append(leaf)
        level_nodes.append(leaf)

    level = 0
    while level < max_levels and len(level_nodes) > 1:
        labels = _cluster(rag.embed([n["text"] for n in level_nodes], cfg), cluster_size)
        groups = defaultdict(list)
        for node, lab in zip(level_nodes, labels):
            groups[lab].append(node)
        if len(groups) >= len(level_nodes):
            break   # clustering didn't compress -> stop
        next_level = []
        for lab, members in sorted(groups.items()):
            summary = _summarize(cfg, [m["text"] for m in members])
            if not summary:
                continue
            sid = "sum_L%d_%s" % (level + 1,
                                  hashlib.sha1(f"{ticker}|{level}|{lab}".encode()).hexdigest()[:10])
            snode = {"node_id": sid, "level": level + 1, "text": summary, "is_summary": True,
                     "child_ids": [m["node_id"] for m in members], "source_file": None, "section": None}
            nodes.append(snode)
            next_level.append(snode)
        level_nodes = next_level
        level += 1
    return nodes


def build_index(cfg: dict, ticker: str, chunks: list[dict], **kw) -> dict:
    """Build the tree, embed every node, and store the collapsed tree in a LanceDB 'raptor' table."""
    import lancedb
    nodes = build_tree(cfg, ticker, chunks, **kw)
    vecs = rag.embed([n["text"] for n in nodes], cfg)
    rows = [{"node_id": n["node_id"], "level": n["level"], "is_summary": n["is_summary"],
             "text": n["text"], "child_ids": json.dumps(n["child_ids"]),
             "source_file": n["source_file"] or "", "section": n["section"] or "", "vector": v}
            for n, v in zip(nodes, vecs)]
    db = lancedb.connect(str(rag._index_path(cfg, ticker)))
    db.create_table("raptor", data=rows, mode="overwrite")
    levels = {}
    for n in nodes:
        levels[n["level"]] = levels.get(n["level"], 0) + 1
    return {"ticker": ticker, "n_nodes": len(nodes), "levels": levels, "leaves": levels.get(0, 0)}


def leaf_chunk_ids(nodes_by_id: dict, node_id: str) -> list[str]:
    """Resolve a (possibly summary) node to its underlying LEAF chunk ids, for grounded citation.
    Orphaned child references (missing from nodes_by_id) are logged, not silently dropped."""
    node = nodes_by_id.get(node_id)
    if not node or not node.get("is_summary"):
        return [node_id] if node_id in nodes_by_id else []
    out: list[str] = []
    for cid in node.get("child_ids", []):
        if cid not in nodes_by_id:
            logging.getLogger(__name__).warning("raptor: orphaned child_id %r under node %r", cid, node_id)
            continue
        out.extend(leaf_chunk_ids(nodes_by_id, cid))
    return out


def retrieve(cfg: dict, ticker: str, query: str, *, k: int = 6) -> dict:
    """Collapsed-tree retrieval over the RAPTOR index: returns nodes across abstraction levels.
    Summary hits carry their child ids so callers can expand to leaves for citation."""
    import lancedb
    qv = rag.embed([query], cfg)[0]
    tbl = lancedb.connect(str(rag._index_path(cfg, ticker))).open_table("raptor")
    hits = tbl.search(qv).metric("cosine").limit(k).to_list()
    results = [{"node_id": h["node_id"], "level": h["level"], "is_summary": h["is_summary"],
                "similarity": round(1.0 - h["_distance"], 3), "text": h["text"],
                "child_ids": json.loads(h.get("child_ids") or "[]"),
                "source_file": h.get("source_file"), "section": h.get("section")} for h in hits]
    return {"query": query, "results": results}
