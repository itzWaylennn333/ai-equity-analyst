"""Company registry: load and merge global defaults + per-company config.

`load_company(ticker)` returns the same cfg dict shape the existing modules
consume (company / data / peers / valuation / scenarios / output), by deep-merging
companies/<TICKER>.yaml over the global config.yaml defaults. If no company file
exists, a minimal config is synthesized so the platform can analyze ANY ticker
(identity + CIK are resolved from live data at pull time; assumptions auto-derive).
"""
from __future__ import annotations

import copy
from pathlib import Path

import utils

COMPANIES_DIR = utils.PROJECT_ROOT / "companies"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins; lists/scalars replace)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_ticker(obj, ticker: str):
    if isinstance(obj, str):
        return obj.replace("{ticker}", ticker)
    if isinstance(obj, dict):
        return {k: _resolve_ticker(v, ticker) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_ticker(v, ticker) for v in obj]
    return obj


def list_companies() -> list[str]:
    if not COMPANIES_DIR.exists():
        return []
    return sorted(p.stem for p in COMPANIES_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def has_company_file(ticker: str) -> bool:
    return (COMPANIES_DIR / f"{ticker.upper()}.yaml").exists()


def load_company(ticker: str) -> dict:
    """Build the merged cfg for `ticker`. Works with or without a company file."""
    ticker = ticker.upper()
    g = utils.load_config("config.yaml")  # global defaults

    cfile = COMPANIES_DIR / f"{ticker}.yaml"
    if cfile.exists():
        company_cfg = utils.load_yaml(cfile)
    else:
        company_cfg = {
            "company": {"name": ticker, "ticker": ticker, "cik": None,
                        "currency": "USD", "sector_template": "standard"},
            "peers": {"core": [], "anchors": []},
        }

    cfg: dict = {}
    cfg["data"] = g["data"]
    cfg["erp"] = g.get("erp", {})
    cfg["sector_templates"] = g.get("sector_templates", {})
    cfg["company"] = company_cfg.get("company", {"ticker": ticker, "name": ticker})
    cfg["company"].setdefault("ticker", ticker)
    cfg["company"].setdefault("sector_template", "standard")
    cfg["peers"] = company_cfg.get("peers", {"core": [], "anchors": []})
    cfg["valuation"] = _deep_merge(g["defaults"]["valuation"], company_cfg.get("valuation", {}))
    cfg["scenarios"] = company_cfg.get("scenarios", g["defaults"].get("scenarios"))
    cfg["output"] = _resolve_ticker(_deep_merge(g["output"], company_cfg.get("output", {})), ticker)
    return cfg


def detect_sector_template(info: dict) -> str:
    """Map a yfinance sector/industry to a valuation template (integrity guardrail).

    Payment/credit-services names (e.g. PYPL) stay 'standard' (DCF applies); banks,
    insurers, asset managers route to 'financials' (unsupported); REITs to 'reit'.
    """
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    if sector == "real estate" or "reit" in industry:
        return "reit"
    fin_kw = ("bank", "insurance", "insurer", "capital markets", "asset management",
              "mortgage", "financial conglomerate", "financial data")
    if any(k in industry for k in fin_kw):
        return "financials"
    return "standard"


def scaffold_company(ticker: str, name: str | None = None, cik: str | None = None,
                     peers: list[str] | None = None) -> Path:
    """Write a minimal companies/<TICKER>.yaml the user can refine."""
    ticker = ticker.upper()
    cik_str = f'"{cik}"' if cik else "null"
    lines = [
        f'# Company config: {name or ticker} ({ticker}) -- auto-scaffolded; refine as needed.',
        "company:",
        f'  name: "{name or ticker}"',
        f'  ticker: "{ticker}"',
        f'  cik: {cik_str}',
        '  currency: "USD"',
        '  sector_template: "standard"',
        "peers:",
    ]
    if peers:
        lines.append("  core:")
        for p in peers:
            lines.append(f'    - {{ ticker: "{p}", name: "{p}", rationale: "" }}')
    else:
        lines.append("  core: []")
    lines.append("  anchors: []")
    lines.append("# valuation & scenarios omitted -> auto-derived from history (see src/assumptions.py)")
    path = COMPANIES_DIR / f"{ticker}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
