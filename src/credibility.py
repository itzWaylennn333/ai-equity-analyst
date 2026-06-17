"""Credibility & noise layer (platform Layer 2b) -- signal vs. noise + source credibility.

Sits downstream of the sentiment engine (Layer 2a). It does NOT score tone; it decides
*how much to trust* each already-scored node, then emits a credibility-weighted aggregate
plus a manipulation / noise-risk flag and the included / excluded node lists. Deterministic
Python only -- every weight traces to a stated rule (integrity rule: no black-box trust).

IMPLEMENTED NOW (works on any ingested document chunk):
  1. Quality gate -- language ID, relevance gate, low-info/spam heuristic, near-duplicate
     dedup (k-shingle Jaccard; swap in datasketch MinHash+LSH for very large corpora).
  2. Source credibility -- per-source-type trust (filings/calls > news > social; reused from
     the sentiment engine), with optional per-domain news seeds.
  4. Aggregation -- credibility-weighted mean tone (weight = quality x source-credibility x
     polarity) with the driving / excluded nodes and a noise-risk flag.

SCAFFOLDED (need a SOCIAL source -- posts with account + timestamp + follower graph -- which
the document pipeline does not produce yet; wire in when StockTwits/Reddit ingestion lands):
  2b. account_credibility = track-record (past calls vs realized returns) x bot/metadata
      discount x centrality prior.
  3.  coordination = account-similarity network (shared cashtags/links/near-identical text/
      synchronized timing) + market-coupled price/volume/chatter spike.
These are explicit extension points (NotImplementedError), and the layer runs fully without
them -- `social_posts=None` simply reports the social layer as unavailable. See
docs/ARCHITECTURE.md section 5. Production upgrades noted inline: fastText `lid.176` for real
language ID, `datasketch` for MinHash+LSH dedup, `crowd-kit` Dawid-Skene/MACE for Bayesian
truth-discovery once multiple raters per node exist.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# Reuse the source-type trust weights defined by the sentiment engine; config may override.
try:
    from sentiment import SOURCE_WEIGHTS as _DEFAULT_SOURCE_WEIGHTS
except Exception:  # pragma: no cover - sentiment is always importable in-package
    _DEFAULT_SOURCE_WEIGHTS = {"filing": 1.0, "transcript": 0.9, "presentation": 0.7,
                               "news": 0.6, "model": 0.3, "social": 0.4, "other": 0.5}

# Tiny English stopword set for a dependency-free language heuristic.
# Production: fastText `lid.176` for real multilingual ID (cfg credibility.lang_model).
_EN_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "was",
            "were", "be", "as", "by", "with", "that", "this", "it", "on", "at", "from",
            "we", "our", "its", "has", "have", "will", "not", "their", "which"}
_FIN_TERMS = ("revenue", "margin", "earnings", "guidance", "cash flow", "operating", "growth",
              "ebitda", "eps", "segment", "quarter", "fiscal", "customers", "volume", "demand")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _config(cfg: dict) -> dict:
    c = dict(cfg.get("credibility") or {})
    c.setdefault("enabled", False)
    c.setdefault("min_relevance", 0.0)      # 0 keeps all document chunks; raise for noisy social
    c.setdefault("dedup_jaccard", 0.85)     # near-duplicate threshold
    c.setdefault("min_alpha_ratio", 0.55)   # below = low-info (XBRL tag / number soup)
    c.setdefault("min_words", 25)
    sw = dict(_DEFAULT_SOURCE_WEIGHTS)
    sw.update(c.get("source_weights") or {})
    c["source_weights"] = sw
    c.setdefault("news_source_seeds", {})   # e.g. {"reuters.com": 0.9, "seekingalpha.com": 0.5}
    return c


# --------------------------------------------------------------------------- #
# 1. Quality gate
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    """Dependency-free English heuristic via stopword density. Production: fastText lid.176."""
    toks = [t.lower() for t in _WORD.findall(text or "")]
    if len(toks) < 8:
        return "und"   # too short to judge language; let the length/info gates handle it
    hits = sum(t in _EN_STOP for t in toks)
    return "en" if hits / len(toks) >= 0.06 else "non-en"


def relevance(text: str, ticker: str, aliases=()) -> float:
    """On-topic score in [0,1]: company/cashtag mention (dominant) + financial-term density."""
    low = (text or "").lower()
    names = {ticker.lower(), f"${ticker.lower()}"} | {a.lower() for a in aliases if a}
    mentions = any(n and n in low for n in names)
    words = max(len((text or "").split()), 1)
    fin = sum(low.count(t) for t in _FIN_TERMS)
    return round(min(1.0, (0.6 if mentions else 0.0) + min(fin / words * 8, 1.0) * 0.4), 3)


def _alpha_ratio(text: str) -> float:
    """Share of non-space chars that are alphabetic. Low = number/tag/URL soup (e.g. XBRL)."""
    nonspace = [c for c in (text or "") if not c.isspace()]
    if not nonspace:
        return 0.0
    return sum(c.isalpha() for c in nonspace) / len(nonspace)


def _shingles(text: str, k: int = 5) -> frozenset:
    toks = [t.lower() for t in _WORD.findall(text or "")]
    if len(toks) < k:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _find_duplicates(chunks: list[dict], threshold: float) -> list[int | None]:
    """Greedy near-dup grouping via k-shingle Jaccard. O(n^2): fine for hundreds of chunks;
    swap in datasketch MinHash+LSH for very large corpora. Returns dup_of[j]=i (j duplicates i)."""
    shings = [_shingles(c.get("text", "")) for c in chunks]
    dup_of: list[int | None] = [None] * len(chunks)
    for i in range(len(chunks)):
        if dup_of[i] is not None:
            continue
        for j in range(i + 1, len(chunks)):
            if dup_of[j] is None and _jaccard(shings[i], shings[j]) >= threshold:
                dup_of[j] = i
    return dup_of


def quality_gate(chunks: list[dict], cfg: dict, ticker: str, aliases=()) -> list[dict]:
    """Per-chunk quality annotation: language, relevance, info density, dedup -> keep + weight."""
    c = _config(cfg)
    dup_of = _find_duplicates(chunks, c["dedup_jaccard"])
    out = []
    for idx, ch in enumerate(chunks):
        text = ch.get("text", "")
        lang, rel, alpha = detect_language(text), relevance(text, ticker, aliases), _alpha_ratio(text)
        nwords = ch.get("n_words") or len(text.split())
        reasons = []
        if lang == "non-en":
            reasons.append("non-english")
        if rel < c["min_relevance"]:
            reasons.append("off-topic")
        if alpha < c["min_alpha_ratio"]:
            reasons.append("low-info")
        if nwords < c["min_words"]:
            reasons.append("too-short")
        if dup_of[idx] is not None:
            reasons.append("duplicate")
        keep = not reasons
        quality = 0.0 if not keep else round(min(1.0, 0.5 + 0.3 * rel + 0.2 * min(alpha / 0.8, 1.0)), 3)
        out.append({"chunk_id": ch.get("chunk_id"), "keep": keep, "reasons": reasons,
                    "lang": lang, "relevance": rel, "alpha_ratio": round(alpha, 3),
                    "duplicate_of": (chunks[dup_of[idx]].get("chunk_id") if dup_of[idx] is not None else None),
                    "quality": quality})
    return out


# --------------------------------------------------------------------------- #
# 2. Source credibility
# --------------------------------------------------------------------------- #
def source_credibility(chunk: dict, cfg: dict) -> float:
    """Trust weight for a node by source type, with optional per-domain news seeds."""
    c = _config(cfg)
    dt = chunk.get("doc_type", "other")
    base = c["source_weights"].get(dt, c["source_weights"].get("other", 0.5))
    if dt == "news" and c["news_source_seeds"]:
        src = (chunk.get("source_file") or "").lower()
        for domain, w in c["news_source_seeds"].items():
            if domain.lower() in src:
                return round(float(w), 3)
    return round(base, 3)


# --------------------------------------------------------------------------- #
# 2b / 3. Social-only layers (scaffolded extension points)
# --------------------------------------------------------------------------- #
def account_credibility(posts: list[dict], cfg: dict) -> dict:
    """track-record (past calls vs realized returns) x bot/metadata discount x centrality prior.
    Requires social posts with {account, ts, ...}. Wire in a StockTwits/Reddit source first."""
    raise NotImplementedError(
        "account_credibility needs a social ingestion source (accounts + timestamps); not wired yet")


def coordination(posts: list[dict], cfg: dict) -> dict:
    """Account-similarity network (shared traces, synchronized timing) + market-coupled spike."""
    raise NotImplementedError(
        "coordination needs social posts with account + timing; not wired yet")


def _social_layers(social_posts, cfg: dict) -> dict:
    if not social_posts:
        return {"available": False, "coordination": {"flagged": False},
                "note": ("account track-record, bot discount, and coordination detection require a "
                         "social source (posts with account + timestamp); none supplied. "
                         "See docs/ARCHITECTURE.md section 5.")}
    return {"available": True, "implemented": False, "coordination": {"flagged": False},
            "note": ("social posts supplied, but account_credibility()/coordination() are not "
                     "implemented yet -- fill those stubs to activate layers 2b & 3.")}


# --------------------------------------------------------------------------- #
# 4. Aggregation + manipulation risk
# --------------------------------------------------------------------------- #
def _direction(tone: float) -> str:
    if tone >= 0.15:
        return "bullish"
    if tone <= -0.15:
        return "bearish"
    return "neutral / mixed"


def manipulation_risk(chunks: list[dict], quality: dict, social: dict, cfg: dict) -> dict:
    """Transparent, rule-based noise/manipulation flag. Strongest once social data flows in."""
    reasons, score = [], 0.0
    if quality["duplicate_rate"] >= 0.30:
        reasons.append(f"high near-duplicate rate ({quality['duplicate_rate']:.0%}) -- "
                       "possible copy-paste / coordinated content")
        score += quality["duplicate_rate"]
    dts = Counter(ch.get("doc_type", "other") for ch in chunks)
    if chunks:
        top_dt, top_n = dts.most_common(1)[0]
        if len(chunks) >= 5 and top_n / len(chunks) >= 0.8 \
                and source_credibility({"doc_type": top_dt}, cfg) < 0.6:
            reasons.append(f"signal concentrated in low-credibility source '{top_dt}' "
                           f"({top_n}/{len(chunks)})")
            score += 0.3
    if social.get("coordination", {}).get("flagged"):
        reasons.append("coordinated-account cluster detected")
        score += 0.5
    level = "high" if score >= 0.6 else "elevated" if score >= 0.3 else "low"
    return {"level": level, "score": round(min(score, 1.0), 3),
            "reasons": reasons or ["no manipulation indicators in available data"]}


def assess(cfg: dict, ticker: str, chunks: list[dict], scores: list[dict],
           *, aliases=(), social_posts=None) -> dict:
    """Run Layer 2b over already-scored chunks -> credibility-weighted aggregate + risk flag.

    `scores[i]` aligns with `chunks[i]` and must carry at least {"tone": float} (and "polar").
    Returns a self-contained sub-report; never raises on bad nodes (degrades to neutral).
    """
    gate = quality_gate(chunks, cfg, ticker, aliases)
    num = den = 0.0
    weighted = []
    for i, (ch, sc, g) in enumerate(zip(chunks, scores, gate)):
        if not g["keep"]:
            continue
        srcw = source_credibility(ch, cfg)
        w = g["quality"] * srcw * max(float(sc.get("polar", 1)), 0.0)
        if w <= 0:
            continue
        num += float(sc.get("tone", 0.0)) * w
        den += w
        weighted.append({"chunk_id": g["chunk_id"], "source_file": ch.get("source_file"),
                         "doc_type": ch.get("doc_type"), "section": ch.get("section"),
                         "tone": round(float(sc.get("tone", 0.0)), 3), "weight": round(w, 3),
                         "source_cred": srcw, "quality": g["quality"]})
    wtone = round(num / den, 3) if den else 0.0
    tones = np.array([w["tone"] for w in weighted], dtype=float)
    disp = float(np.std(tones)) if len(tones) > 1 else 1.0
    conviction = float(np.clip(abs(wtone) / (disp + 0.1) * min(len(tones) / 20, 1.0), 0, 1))

    n = len(chunks)
    rate = lambda reason: round(sum(reason in g["reasons"] for g in gate) / n, 3) if n else 0.0
    quality = {"kept": sum(g["keep"] for g in gate), "dropped": sum(not g["keep"] for g in gate),
               "duplicate_rate": rate("duplicate"), "low_info_rate": rate("low-info"),
               "off_topic_rate": rate("off-topic"), "non_english_rate": rate("non-english")}
    social = _social_layers(social_posts, cfg)
    weighted.sort(key=lambda x: -x["weight"])
    return {
        "enabled": True, "ticker": ticker, "n_nodes": n,
        "weighted_tone": wtone, "weighted_conviction": round(conviction, 2),
        "direction": _direction(wtone),
        "quality": quality,
        "by_source_credibility": {dt: source_credibility({"doc_type": dt}, cfg)
                                  for dt in sorted({ch.get("doc_type", "other") for ch in chunks})},
        "top_weighted": weighted[:5],
        "excluded": [{"chunk_id": g["chunk_id"], "reasons": g["reasons"]}
                     for g in gate if not g["keep"]][:50],
        "manipulation_risk": manipulation_risk(chunks, quality, social, cfg),
        "social": social,
    }
