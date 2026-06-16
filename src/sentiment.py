"""Sentiment engine (platform Layer 2a) -- DIRECTIONAL financial sentiment.

Tiered by what's available; all tiers return the same shape so callers don't care:
  1. lexicon  -- Loughran-McDonald-style finance lexicon. No heavy deps; always works.
  2. finbert  -- ProsusAI/finbert via transformers (optional; better on news/headlines).
  3. llm      -- local Ollama-compatible endpoint for graded sentiment + open aspects
                 (optional; ideal on the DGX Spark).

Output is a SIGNED, aspect-decomposed signal with CONVICTION -- not just polarity --
because a view needs direction + strength, and tone must be attributed to topics
(margins, guidance, litigation, ...) rather than averaged into mush.

Honest limits (see docs/RESEARCH.md A): lexicon tone is coarse and misses negation/
sarcasm; the bundled lexicon is a curated SEED, not the full (licensed) L-M dictionary
-- point `sentiment.lexicon_path` at the real CSV for production use. Sentiment is a
complementary signal, weighted below filings/calls, never a standalone trade.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

import utils

# --------------------------------------------------------------------------- #
# Curated finance SEED lexicon (representative, not the full Loughran-McDonald set).
# Replace by pointing cfg['sentiment']['lexicon_path'] at the L-M master CSV.
# --------------------------------------------------------------------------- #
SEED_LEXICON = {
    "positive": {
        "growth", "growing", "grew", "increase", "increased", "increases", "increasing",
        "gain", "gains", "gained", "profit", "profitable", "profitability", "strong",
        "strength", "strengthened", "improve", "improved", "improvement", "improving",
        "favorable", "record", "exceeded", "exceed", "outperform", "outperformed",
        "robust", "accelerate", "accelerated", "acceleration", "expansion", "expanding",
        "efficient", "efficiency", "momentum", "upside", "beat", "beats", "tailwind",
        "resilient", "resilience", "leadership", "leading", "premium", "rebound",
        "recovery", "recovered", "higher", "surpassed", "optimistic", "upgrade",
    },
    "negative": {
        "loss", "losses", "decline", "declined", "declines", "declining", "decrease",
        "decreased", "decreasing", "weak", "weakness", "weakened", "weaker", "adverse",
        "adversely", "impair", "impaired", "impairment", "deficit", "deficiency",
        "downturn", "restructuring", "restructure", "litigation", "lawsuit", "default",
        "bankruptcy", "insolvency", "write-down", "writedown", "writeoff", "shortfall",
        "headwind", "headwinds", "pressure", "pressured", "erosion", "eroding", "eroded",
        "miss", "missed", "disappointing", "disappoint", "deteriorate", "deteriorated",
        "deterioration", "slowdown", "slowing", "challenging", "challenged", "lower",
        "underperform", "underperformed", "downgrade", "recall", "fraud", "breach",
        "delay", "delayed", "concern", "concerns", "soft", "softness", "contraction",
    },
    "uncertainty": {
        "may", "might", "could", "risk", "risks", "uncertain", "uncertainty", "approximate",
        "approximately", "volatile", "volatility", "fluctuate", "fluctuation", "exposure",
        "contingent", "possible", "possibly", "unpredictable", "depend", "depends",
        "pending", "subject", "assumption", "assumptions", "estimate", "estimates",
    },
    "litigious": {
        "litigation", "lawsuit", "lawsuits", "plaintiff", "defendant", "settlement",
        "regulatory", "regulation", "investigation", "subpoena", "indictment", "fine",
        "fines", "penalty", "penalties", "allegation", "alleged", "complaint", "injunction",
    },
}

# Aspect -> trigger keywords (substring match on tokens/sentences).
ASPECTS = {
    "margins": ["margin", "gross margin", "operating margin", "profitability", "cost of revenue"],
    "guidance": ["guidance", "outlook", "forecast", "expect", "anticipate", "full-year", "full year"],
    "growth": ["growth", "revenue growth", "expansion", "accelerat", "top-line", "top line"],
    "demand": ["demand", "volume", "traffic", "orders", "bookings", "backlog", "engagement"],
    "costs": ["cost", "expense", "inflation", "opex", "headcount", "restructuring", "spend"],
    "competition": ["competit", "market share", "rivals", "pricing pressure", "share loss"],
    "liquidity": ["liquidity", "leverage", "debt", "covenant", "refinanc", "free cash flow", "buyback", "dividend"],
    "litigation": ["litigation", "lawsuit", "regulatory", "investigation", "settlement", "fine"],
}

# Source-type trust weights (filings/calls > news > social) -- research B/A.
SOURCE_WEIGHTS = {"filing": 1.0, "transcript": 0.9, "presentation": 0.7,
                  "news": 0.6, "model": 0.3, "social": 0.4, "other": 0.5}

_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def load_lexicon(cfg: dict) -> dict:
    """Full Loughran-McDonald CSV if configured & present, else the curated seed."""
    path = (cfg.get("sentiment") or {}).get("lexicon_path")
    if path and utils.resolve(path).exists():
        import pandas as pd
        df = pd.read_csv(utils.resolve(path))
        cols = {c.lower(): c for c in df.columns}
        word_col = cols.get("word") or df.columns[0]
        words = df[word_col].astype(str).str.lower()
        lex = {}
        for cat in ("negative", "positive", "uncertainty", "litigious"):
            if cat in cols:  # L-M stores a year (>0) when the word is in that category
                flag = df[cols[cat]]
                lex[cat] = set(words[flag.astype(float) > 0])
        if lex:
            return lex
    return {k: set(v) for k, v in SEED_LEXICON.items()}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def score_text_lexicon(text: str, lex: dict) -> dict:
    toks = _tokens(text)
    n = max(len(toks), 1)
    cnt = {cat: sum(t in words for t in toks) for cat, words in lex.items()}
    pos, neg = cnt.get("positive", 0), cnt.get("negative", 0)
    tone = (pos - neg) / (pos + neg) if (pos + neg) else 0.0   # signed in [-1, 1]
    return {"tone": tone, "pos": pos, "neg": neg, "n_words": len(toks),
            "uncertainty": cnt.get("uncertainty", 0) / n, "litigious": cnt.get("litigious", 0) / n,
            "polar": pos + neg}


# --------------------------------------------------------------------------- #
# Optional adapters (lazy; only import heavy deps when used)
# --------------------------------------------------------------------------- #
def score_texts_finbert(texts: list[str]) -> list[dict]:
    from transformers import pipeline  # type: ignore
    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    out = []
    for r in clf(list(texts)):
        lbl = r["label"].lower()
        sign = {"positive": 1, "negative": -1}.get(lbl, 0)
        out.append({"tone": sign * float(r["score"]), "label": lbl, "score": float(r["score"]),
                    "pos": int(sign > 0), "neg": int(sign < 0), "polar": int(sign != 0)})
    return out


# JSON Schema for Ollama structured outputs -- constrains decoding to valid,
# parseable JSON. Far more robust than plain {"format": "json"}, which some
# reasoning models silently ignore (e.g. the Qwen3.x think:false interaction).
_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "tone": {"type": "number"},
        "conviction": {"type": "number"},
        "aspects": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"aspect": {"type": "string"}, "tone": {"type": "number"}},
                      "required": ["aspect", "tone"]},
        },
        "rationale": {"type": "string"},
    },
    "required": ["tone", "conviction", "aspects", "rationale"],
}


def _first_json_object(s: str) -> str | None:
    """First balanced top-level {...} substring (string/escape aware), or None."""
    depth, start = 0, -1
    in_str = esc = False
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _parse_llm_json(raw: str) -> dict:
    """Best-effort JSON from a model reply, tolerant of 'thinking' models.

    Strips <think>...</think> spans, then tries the whole string, then the first
    balanced object. Raises ValueError if nothing parses.
    """
    import json
    if not raw:
        raise ValueError("empty response")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for candidate in (cleaned, _first_json_object(cleaned)):
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                continue
    raise ValueError("no parseable JSON object in response")


def score_text_llm(text: str, endpoint: str, model: str, *, retries: int = 1) -> dict:
    """Graded sentiment + aspects via an Ollama-compatible /api/chat endpoint.

    Hardened for reasoning/'thinking' models: requests schema-constrained JSON,
    reads message.content (falling back to message.thinking), strips <think>
    blocks, and extracts the first balanced JSON object. A chunk that still won't
    parse after one retry degrades to a neutral, non-polar score (polar=0) so a
    single bad response can't abort or skew the whole run -- this never raises.
    """
    import requests
    prompt = ("You are a sell-side equity analyst. Read the excerpt and return STRICT JSON: "
              '{"tone": <float -1..1, bearish..bullish>, "conviction": <float 0..1>, '
              '"aspects": [{"aspect": str, "tone": float}], "rationale": str}. Excerpt:\n\n' + text[:4000])
    payload = {"model": model, "format": _LLM_SCHEMA, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{endpoint.rstrip('/')}/api/chat", timeout=120, json=payload)
            r.raise_for_status()
            msg = r.json().get("message", {}) or {}
            data = _parse_llm_json(msg.get("content") or msg.get("thinking") or "")
            data["tone"] = float(np.clip(data.get("tone", 0.0), -1.0, 1.0))
            data["polar"] = 1
            return data
        except Exception as e:  # network, HTTP, or parse failure -> retry then degrade
            last_err = e
    return {"tone": 0.0, "polar": 0, "conviction": 0.0, "aspects": [],
            "rationale": f"llm parse/call failed: {last_err}", "error": True}


# --------------------------------------------------------------------------- #
# Aspect attribution + directional aggregation
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.split()) >= 4]


def aspect_sentiment(chunks: list[dict], lex: dict) -> dict:
    agg = {a: {"tone_sum": 0.0, "wt": 0.0, "mentions": 0} for a in ASPECTS}
    for ch in chunks:
        for sent in _split_sentences(ch.get("text", "")):
            low = sent.lower()
            for aspect, kws in ASPECTS.items():
                if any(k in low for k in kws):
                    sc = score_text_lexicon(sent, lex)
                    if sc["polar"]:
                        agg[aspect]["tone_sum"] += sc["tone"] * sc["polar"]
                        agg[aspect]["wt"] += sc["polar"]
                    agg[aspect]["mentions"] += 1
    out = {}
    for a, d in agg.items():
        if d["mentions"]:
            out[a] = {"tone": (d["tone_sum"] / d["wt"]) if d["wt"] else 0.0, "mentions": d["mentions"]}
    return out


def _direction_label(tone: float) -> str:
    if tone >= 0.15:
        return "bullish"
    if tone <= -0.15:
        return "bearish"
    return "neutral / mixed"


def analyze(cfg: dict, ticker: str, chunks: list[dict], method: str | None = None,
            prior_tone: float | None = None) -> dict:
    """Directional sentiment report over a set of (ingested) chunks."""
    scfg = cfg.get("sentiment") or {}
    method = method or scfg.get("method", "lexicon")
    lex = load_lexicon(cfg)
    if not chunks:
        return {"method": method, "n_chunks": 0, "overall_tone": None,
                "direction": "no data", "by_aspect": {}, "by_source": {}}

    texts = [c.get("text", "") for c in chunks]
    if method == "finbert":
        scores = score_texts_finbert(texts)
    elif method == "llm":
        ep, mdl = scfg.get("llm_endpoint", "http://localhost:11434"), scfg.get("llm_model", "qwen2.5:32b")
        scores = [score_text_llm(t, ep, mdl) for t in texts]
    else:
        scores = [score_text_lexicon(t, lex) for t in texts]

    # Source-weighted, polarity-weighted overall tone.
    num = den = 0.0
    by_src = defaultdict(lambda: {"tone_sum": 0.0, "wt": 0.0, "n": 0})
    tones = []
    for ch, sc in zip(chunks, scores):
        w = SOURCE_WEIGHTS.get(ch.get("doc_type", "other"), 0.5) * max(sc.get("polar", 1), 0.0001)
        num += sc["tone"] * w
        den += w
        tones.append(sc["tone"])
        s = by_src[ch.get("doc_type", "other")]
        s["tone_sum"] += sc["tone"] * w; s["wt"] += w; s["n"] += 1
    overall = num / den if den else 0.0

    # Conviction: scaled by signal strength vs dispersion and sample size.
    arr = np.array(tones, dtype=float)
    polar_share = float(np.mean([s.get("polar", 0) > 0 for s in scores]))
    dispersion = float(np.std(arr)) if len(arr) > 1 else 1.0
    conviction = float(np.clip(abs(overall) / (dispersion + 0.1) * min(len(arr) / 20, 1.0) * (0.5 + polar_share / 2), 0, 1))

    report = {
        "method": method, "ticker": ticker, "n_chunks": len(chunks),
        "overall_tone": round(overall, 3), "conviction": round(conviction, 2),
        "direction": _direction_label(overall),
        "by_aspect": {a: {"tone": round(v["tone"], 3), "mentions": v["mentions"]}
                      for a, v in aspect_sentiment(chunks, lex).items()},
        "by_source": {k: round(v["tone_sum"] / v["wt"], 3) for k, v in by_src.items() if v["wt"]},
        "uncertainty_rate": round(float(np.mean([s.get("uncertainty", 0) for s in scores])), 4),
    }
    if prior_tone is not None:
        report["delta_vs_prior"] = round(overall - prior_tone, 3)  # tone CHANGE = the directional signal
    # Most-positive / most-negative excerpts for transparency.
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1]["tone"])
    report["top_negative"] = [{"section": c.get("section"), "tone": round(s["tone"], 2),
                               "text": c.get("text", "")[:240]} for c, s in ranked[:2] if s["tone"] < 0]
    report["top_positive"] = [{"section": c.get("section"), "tone": round(s["tone"], 2),
                               "text": c.get("text", "")[:240]} for c, s in ranked[::-1][:2] if s["tone"] > 0]
    return report
