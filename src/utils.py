"""Shared helpers: config loading, disk caching, and a provenance manifest.

Design goals (for a defensible, reproducible research artifact):
- Config is the single source of truth (`config.yaml`).
- Every raw pull is cached to `data/raw/` so the analysis reproduces offline and
  we never re-hit a data source unnecessarily.
- Every pull is logged to `data/raw/_manifest.json` with its source, identifier,
  URL/endpoint and a UTC timestamp, so any figure's origin is auditable.
"""
from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> dict:
    """Load the YAML config. Relative paths resolve from the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml(path: str | Path) -> dict:
    """Load any YAML file (absolute or project-relative path)."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path to an absolute Path under the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def cache_is_fresh(path: str | Path, max_age_days: float) -> bool:
    """True if `path` exists and is younger than `max_age_days`."""
    p = resolve(path)
    if not p.exists():
        return False
    mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime)
    age_days = (_dt.datetime.now() - mtime).total_seconds() / 86400.0
    return age_days <= max_age_days


# --------------------------------------------------------------------------- #
# Disk I/O (human-readable formats so the cache is auditable)
# --------------------------------------------------------------------------- #
def save_df(df: pd.DataFrame, path: str | Path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p)
    return p


def load_df(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(resolve(path), index_col=0, **kwargs)


def save_json(obj: Any, path: str | Path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def load_json(path: str | Path) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Provenance manifest
# --------------------------------------------------------------------------- #
def record_provenance(
    manifest_path: str | Path,
    source: str,
    identifier: str,
    url: str,
    files: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Append/replace an entry in the provenance manifest.

    Keyed by ``source:identifier`` (e.g. ``yahoo:PYPL``, ``edgar:0001633917``)
    so re-pulls overwrite the prior record with a fresh timestamp.
    """
    mp = resolve(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if mp.exists():
        try:
            manifest = load_json(mp)
        except Exception:
            manifest = {}
    key = f"{source}:{identifier}"
    manifest[key] = {
        "source": source,
        "identifier": identifier,
        "url": url,
        "files": files or [],
        "retrieved_utc": utcnow_iso(),
        **({"extra": extra} if extra else {}),
    }
    save_json(manifest, mp)


_CCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
                "HKD": "HK$", "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "INR": "₹",
                "KRW": "₩", "BRL": "R$", "TWD": "NT$"}


def currency_symbol(code: str | None) -> str:
    """Display symbol for a currency code (defaults to the code + space, '$' for USD)."""
    if not code:
        return "$"
    return _CCY_SYMBOLS.get(str(code).upper(), f"{code} ")


def fmt_money(x: float, unit: str = "B") -> str:
    """Format a raw dollar figure into $B (default) or $M for display."""
    if x is None:
        return "n/a"
    div = {"B": 1e9, "M": 1e6, "K": 1e3}[unit]
    return f"${x / div:,.2f}{unit}"
