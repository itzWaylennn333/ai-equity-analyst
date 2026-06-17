"""Reconciliation agent (platform Layer 3 / P6) -- cross-check filing text vs computed numbers.

The integrity backbone of the agentic layer: a figure the LLM extracts from the *filing text*
(via the RAG extractor, with a verified citation) is compared against the *deterministic* value
the platform already computes from SEC EDGAR / yfinance (`financials.build_historical`). Agreement
corroborates the number; disagreement is surfaced as a conflict to investigate -- never silently
reconciled. This is the architecture's "reconciliation agent: API vs filing vs upload conflicts"
(docs/ARCHITECTURE.md s5/s8) and enforces "no fabricated numbers; every figure traces to a source".

Deterministic Python owns the comparison (integrity rule). Filing figures are reported at varying
unit scales (a 10-K says "33,172" meaning $33,172 million; EDGAR stores 33,172,000,000), so the
match is scale-robust: two values agree if they coincide within tolerance at SOME common unit scale.
"""
from __future__ import annotations

import re

import pandas as pd

import utils

# Word-boundary magnitude suffixes (exact trailing token only, so "(basic)" is NOT read as billions).
_SUFFIX = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
           "million": 1e6, "mm": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_NUM = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")   # also matches leading-decimal ".5"
# Unit scales to test when checking whether two magnitudes represent the same quantity.
_SCALES = (1e12, 1e9, 1e6, 1e3, 1e2, 1.0, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12)

# Canonical filing<->deterministic metric map (column in historical_{ticker}.csv -> question template).
_METRICS = {
    "net revenues": ("revenue", "What were {t}'s total net revenues for fiscal year {fy}?"),
    "net income": ("net_income", "What was {t}'s net income (loss) for fiscal year {fy}?"),
    "operating income": ("operating_income", "What was {t}'s operating income for fiscal year {fy}?"),
}


def _parse_number(s) -> float | None:
    """Parse a filing figure string to a signed magnitude. Handles $, commas, (negatives), and
    word suffixes (billion/million/thousand/k/m/b...). Returns None if no number is present."""
    if s is None:
        return None
    txt = str(s).strip().lower()
    neg = txt.startswith("(") and txt.endswith(")")
    stripped = txt.replace("$", "")
    m = _NUM.search(stripped)
    if not m:
        return None
    raw = m.group(0)
    intpart = raw.lstrip("+-").split(".")[0]
    if "," in intpart and not re.fullmatch(r"\d{1,3}(,\d{3})+", intpart):
        return None   # malformed thousands grouping (e.g. "123,45") -> reject, don't silently coerce
    try:
        num = float(raw.replace(",", ""))
    except ValueError:
        return None
    tok = re.match(r"\s*([a-z]+)", stripped[m.end():])   # exact trailing word only
    if tok:
        num *= _SUFFIX.get(tok.group(1), 1.0)
    return -num if neg else num


def _same_magnitude(a: float, b: float, tol: float):
    """Return the unit scale at which a*scale ~= b (within relative tol), else None."""
    if b is None or a is None:
        return None
    denom = max(abs(b), 1e-9)
    for scale in _SCALES:
        if abs(a * scale - b) / denom <= tol:
            return scale
    return None


def reconcile(cfg: dict, ticker: str, items: list[dict], *, tol: float = 0.02, extractor=None) -> dict:
    """Cross-check each item's filing-extracted figure against its deterministic expected value.

    items: [{"metric": str, "question": str, "expected": float|None, "period": any}]
    extractor: callable(cfg, ticker, question)->rag.extract result; defaults to rag.extract
    (injectable so the logic is unit-testable without a live LLM).
    """
    if extractor is None:
        import rag
        extractor = rag.extract
    rows = []
    for it in items:
        out = extractor(cfg, ticker, it["question"])
        rec = {"metric": it["metric"], "period": it.get("period"),
               "expected": it.get("expected"), "filing_value": None, "filing_number": None,
               "status": None, "scale": None, "rel_diff": None, "citation": None,
               "reason": None}
        if not out.get("found"):
            rec["status"] = "filing_not_found"
            rec["reason"] = out.get("reason")
        else:
            rec["filing_value"] = out["answer"]["value"]
            rec["citation"] = out.get("citation")
            fn = _parse_number(rec["filing_value"])
            rec["filing_number"] = fn
            en = it.get("expected")
            if en is None:
                rec["status"] = "no_deterministic"
            elif fn is None:
                rec["status"] = "unparseable_filing_value"
            else:
                scale = _same_magnitude(fn, float(en), tol)
                rec["status"] = "match" if scale is not None else "mismatch"
                rec["scale"] = scale
                if scale is not None:   # rel_diff is only meaningful at a matched unit scale
                    rec["rel_diff"] = round(abs(fn * scale - float(en)) / max(abs(float(en)), 1e-9), 4)
        rows.append(rec)
    counts = {s: sum(r["status"] == s for r in rows)
              for s in ("match", "mismatch", "filing_not_found", "no_deterministic", "unparseable_filing_value")}
    return {"ticker": ticker, "tolerance": tol, "items": rows, "counts": counts,
            "conflicts": [r for r in rows if r["status"] == "mismatch"],
            "ok": counts["mismatch"] == 0}


def _load_financials(cfg: dict, ticker: str) -> pd.DataFrame | None:
    path = utils.resolve(cfg["data"]["processed_dir"]) / f"historical_{ticker}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def reconcile_financials(cfg: dict, ticker: str, *, fin: pd.DataFrame | None = None,
                         fy: int | None = None, tol: float = 0.02, extractor=None) -> dict:
    """Reconcile the latest fiscal year's headline figures (revenue / net income / operating income)
    from the filing against the deterministic historical table written by `run.py`."""
    if fin is None:
        fin = _load_financials(cfg, ticker)
    if fin is None or fin.empty:
        return {"ticker": ticker, "error": "no deterministic historical_{ticker}.csv -- run run.py first",
                "items": [], "counts": {}, "ok": None}
    fy = fy or int(max(int(y) for y in fin.index))
    items, skipped = [], []
    for metric, (col, qtmpl) in _METRICS.items():
        if col not in fin.columns or fy not in fin.index:
            skipped.append({"metric": metric,
                            "reason": "column missing" if col not in fin.columns else "fy not in index"})
            continue
        val = fin.loc[fy, col]
        items.append({"metric": metric, "period": fy,
                      "question": qtmpl.format(t=ticker, fy=fy),
                      "expected": float(val) if pd.notna(val) else None})
    out = reconcile(cfg, ticker, items, tol=tol, extractor=extractor)
    out["skipped"] = skipped
    return out
