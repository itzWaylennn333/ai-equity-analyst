"""Narrative drafting agent (platform Layer 3 / P6) -- grounded, inline-cited section prose.

Drafts a report section from retrieved filing spans where EVERY sentence carries an inline
citation to the passage it is grounded in, then verifies each sentence against its cited passage
(LLM-as-judge) and drops/flags any that are not supported. No claim or figure enters the prose
without a verified source span (integrity rules; docs/ARCHITECTURE.md s4/s8).

Built on the RAG primitives (rag.retrieve + schema-constrained generation). PydanticAI / Outlines
are an optional orchestration layer; the integrity-critical parts (schema output, citation,
grounding verification) are enforced here directly.
"""
from __future__ import annotations

import rag

_DRAFT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"sentences": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"text": {"type": "string"}, "passage": {"type": "integer"}},
        "required": ["text", "passage"]}}},
    "required": ["sentences"],
}
_VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["supported"]}}},
    "required": ["verdicts"],
}


def draft_section(cfg: dict, ticker: str, topic: str, *, k: int = 6,
                  chunks: list[dict] | None = None, verify: bool = True) -> dict:
    """Draft a grounded, inline-cited paragraph for `topic`; unsupported sentences are flagged out.

    Returns {topic, draft (markdown with [n] cites), sentences, sources, flagged_unsupported,
    grounded_ratio}. Abstains (empty draft + note) when retrieval confidence is low.
    """
    ret = rag.retrieve(cfg, ticker, topic, k=k, chunks=chunks)
    passages = ret["results"]
    if not passages or not ret["confident"]:
        return {"topic": topic, "draft": "", "sentences": [], "sources": [], "flagged_unsupported": [],
                "grounded_ratio": 0.0,
                "note": f"insufficient grounded evidence for '{topic}' (top cosine {ret['top_similarity']})"}

    listing = "\n\n".join(f"[{i + 1}] {(p['text'] or '')[:1200]}" for i, p in enumerate(passages))
    prompt = (f"You are an equity analyst drafting the '{topic}' section for {ticker}. Using ONLY the "
              "numbered passages, write 3-6 concise, factual sentences. Each sentence MUST be grounded "
              "in exactly one passage; record that passage number. Do not state any figure or claim not "
              'present in the passages.\n\nReturn JSON {"sentences":[{"text": str, "passage": int}]}.\n\n'
              f"PASSAGES:\n{listing}")
    try:
        drafted = rag._chat_json(cfg, prompt, _DRAFT_SCHEMA).get("sentences", [])
    except Exception as e:
        return {"topic": topic, "draft": "", "sentences": [], "sources": [], "flagged_unsupported": [],
                "grounded_ratio": 0.0, "note": f"draft failed: {e}"}

    sents = [s for s in drafted if isinstance(s.get("passage"), int) and 1 <= s["passage"] <= len(passages)]
    verify_ran = bool(verify and sents)
    verdicts: list = [None] * len(sents)
    if verify_ran:
        vlist = "\n\n".join(
            f"[{j + 1}] SENTENCE: {s['text']}\nCITED PASSAGE: {(passages[s['passage'] - 1]['text'] or '')[:1000]}"
            for j, s in enumerate(sents))
        vprompt = ("For each item decide whether the CITED PASSAGE supports the SENTENCE (the claim is "
                   "stated or directly entailed by that passage, not merely topically related). Return JSON "
                   '{"verdicts":[{"supported": bool, "reason": str}]} in the SAME ORDER as the items.\n\n' + vlist)
        try:
            vv = rag._chat_json(cfg, vprompt, _VERIFY_SCHEMA).get("verdicts", [])
            for j in range(min(len(vv), len(sents))):
                verdicts[j] = vv[j]
        except Exception as e:   # a crashed judge must NOT silently pass as supported
            verdicts = [{"supported": None, "reason": f"verification unavailable: {e}"} for _ in sents]

    out_sents, grounded = [], 0
    for j, s in enumerate(sents):
        v = verdicts[j] if isinstance(verdicts[j], dict) else {}
        supported = v.get("supported")            # True | False | None (unverified)
        if supported is True:
            grounded += 1
        out_sents.append({"text": s["text"], "passage": s["passage"],
                          "chunk_id": passages[s["passage"] - 1]["chunk_id"],
                          "supported": supported, "reason": v.get("reason")})

    # With verification, the clean draft keeps ONLY explicitly-supported sentences; unsupported AND
    # unverified ones are flagged out. Without verification, keep the cited sentences as drafted.
    if verify_ran:
        kept = [s for s in out_sents if s["supported"] is True]
        flagged = [s for s in out_sents if s["supported"] is not True]
    else:
        kept, flagged = out_sents, []
    used: list = []
    src_by_id = {p["chunk_id"]: p for p in passages}

    def _cite(cid: str) -> int:
        if cid not in used:
            used.append(cid)
        return used.index(cid) + 1

    draft_md = " ".join(f"{s['text']} [{_cite(s['chunk_id'])}]" for s in kept)
    sources = [{"n": i + 1, "chunk_id": cid, "source_file": src_by_id[cid].get("source_file"),
                "section": src_by_id[cid].get("section")} for i, cid in enumerate(used)]
    grounded_ratio = (round(grounded / len(out_sents), 2) if (out_sents and verify_ran)
                      else (None if not verify_ran else 0.0))
    return {"topic": topic, "draft": draft_md, "sentences": out_sents, "sources": sources,
            "flagged_unsupported": flagged, "grounded_ratio": grounded_ratio,
            "retrieval": {"top_similarity": ret["top_similarity"], "reranked": ret.get("reranked")}}
