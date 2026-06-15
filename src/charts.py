"""Chart generation. Clean, consistent, report-grade matplotlib figures.

Every chart carries a source footnote so the deliverable is self-documenting.
PNGs are written at 200 DPI to the configured charts directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import utils

# --- palette (restrained, PayPal-ish navy/teal) ---
NAVY = "#003087"
TEAL = "#009cde"
GREY = "#6b7280"
LIGHT = "#cbd5e1"
RED = "#c0392b"
GREEN = "#1e8449"
AMBER = "#d68910"
SRC = "Source: SEC 10-K filings (FY2022-FY2025) & yfinance. Author analysis."


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (8.2, 4.6),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def _footnote(fig, text=SRC):
    fig.text(0.01, -0.04, text, ha="left", va="top", fontsize=7.5, color=GREY)


def _save(fig, cfg, name):
    out = utils.resolve(cfg["output"]["charts_dir"]) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def chart_price_history(cfg, prices, info):
    s = prices["Close"].copy()
    s.index = pd.to_datetime(s.index, utc=True).tz_convert(None)
    fig, ax = plt.subplots()
    ax.plot(s.index, s.values, color=NAVY, lw=1.3)
    hi, lo = info.get("fiftyTwoWeekHigh"), info.get("fiftyTwoWeekLow")
    last = info.get("currentPrice")
    if hi and lo:
        ax.axhspan(lo, hi, color=TEAL, alpha=0.08)
        ax.axhline(hi, color=GREY, ls="--", lw=0.8)
        ax.axhline(lo, color=GREY, ls="--", lw=0.8)
        ax.text(s.index[0], hi, f"  52-wk high ${hi:,.0f}", va="bottom", fontsize=8, color=GREY)
        ax.text(s.index[0], lo, f"  52-wk low ${lo:,.0f}", va="top", fontsize=8, color=GREY)
    if last:
        ax.scatter([s.index[-1]], [last], color=RED, zorder=5)
        ax.text(s.index[-1], last, f" ${last:,.2f}", va="center", color=RED, fontsize=9, fontweight="bold")
    ax.set_title("PYPL share price (6 years)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%d"))
    ax.set_ylabel("Share price")
    _footnote(fig, "Source: yfinance (auto-adjusted close). Author analysis.")
    return _save(fig, cfg, "01_price_history.png")


def chart_revenue_growth(cfg, H):
    d = H.dropna(subset=["revenue"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    ax.bar(yrs, d["revenue"] / 1e9, color=NAVY, width=0.6, label="Net revenue ($B)")
    ax.set_ylabel("Net revenue ($B)")
    ax.set_title("Revenue growth is decelerating")
    for x, v in zip(yrs, d["revenue"] / 1e9):
        ax.text(x, v + 0.3, f"${v:,.1f}B", ha="center", fontsize=8.5, color=NAVY)
    ax2 = ax.twinx()
    ax2.plot(yrs, d["revenue_growth"] * 100, color=AMBER, marker="o", lw=2, label="YoY growth (%)")
    ax2.set_ylabel("YoY revenue growth (%)", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax2.grid(False)
    ax2.set_ylim(0, max(12, d["revenue_growth"].max() * 100 * 1.4))
    for x, g in zip(yrs, d["revenue_growth"] * 100):
        if pd.notna(g):
            ax2.text(x, g + 0.4, f"{g:,.1f}%", ha="center", fontsize=8.5, color=AMBER)
    ax.set_xticks(yrs)
    _footnote(fig)
    return _save(fig, cfg, "02_revenue_growth.png")


def chart_margins(cfg, H):
    d = H.dropna(subset=["revenue"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    series = [("gross_margin", "Gross (transaction) margin", TEAL),
              ("ebitda_margin", "EBITDA margin", NAVY),
              ("operating_margin", "Operating margin (GAAP)", GREEN),
              ("net_margin", "Net margin", AMBER)]
    for col, lbl, c in series:
        ax.plot(yrs, d[col] * 100, marker="o", lw=1.8, color=c, label=lbl)
    ax.set_title("Margins: operating leverage despite gross-margin pressure")
    ax.set_ylabel("Margin (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(yrs)
    ax.legend(fontsize=8, ncol=2, loc="center left")
    _footnote(fig)
    return _save(fig, cfg, "03_margins.png")


def chart_fcf(cfg, H):
    d = H.dropna(subset=["fcf_reported"])
    yrs = d.index.astype(int)
    x = np.arange(len(yrs))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, d["fcf_reported"] / 1e9, width=0.4, color=NAVY, label="Reported FCF (OCF - capex)")
    ax.bar(x + 0.2, d["fcff"] / 1e9, width=0.4, color=TEAL, label="FCFF (SBC expensed, unlevered)")
    ax.set_title("Free cash flow generation is substantial")
    ax.set_ylabel("$B")
    ax.set_xticks(x)
    ax.set_xticklabels(yrs)
    for xi, v in zip(x - 0.2, d["fcf_reported"] / 1e9):
        ax.text(xi, v + 0.1, f"{v:,.1f}", ha="center", fontsize=8, color=NAVY)
    for xi, v in zip(x + 0.2, d["fcff"] / 1e9):
        ax.text(xi, v + 0.1, f"{v:,.1f}", ha="center", fontsize=8, color=TEAL)
    ax.legend(fontsize=8, loc="upper left")
    _footnote(fig)
    return _save(fig, cfg, "04_fcf.png")


def chart_tpv_takerate(cfg, H):
    d = H.dropna(subset=["tpv_usd_b"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    ax.bar(yrs, d["tpv_usd_b"] / 1000, color=LIGHT, width=0.6, label="TPV ($T)")
    ax.set_ylabel("Total Payment Volume ($T)")
    ax.set_title("The core tension: volume up, take-rate down")
    for x, v in zip(yrs, d["tpv_usd_b"] / 1000):
        ax.text(x, v + 0.02, f"${v:,.2f}T", ha="center", fontsize=8.5, color=GREY)
    ax2 = ax.twinx()
    ax2.plot(yrs, d["total_take_rate"] * 100, color=RED, marker="o", lw=2.2, label="Total take rate (%)")
    ax2.set_ylabel("Total take rate (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.grid(False)
    ax2.set_ylim(1.6, 2.1)
    for x, v in zip(yrs, d["total_take_rate"] * 100):
        ax2.text(x, v + 0.01, f"{v:,.2f}%", ha="center", fontsize=8.5, color=RED)
    ax.set_xticks(yrs)
    _footnote(fig)
    return _save(fig, cfg, "05_tpv_takerate.png")


def chart_shares_buybacks(cfg, H):
    d = H.dropna(subset=["diluted_shares"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    ax.bar(yrs, d["buybacks"] / 1e9, color=GREEN, width=0.6, label="Buybacks ($B)")
    ax.set_ylabel("Share repurchases ($B)", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    for x, v in zip(yrs, d["buybacks"] / 1e9):
        ax.text(x, v + 0.05, f"${v:,.1f}B", ha="center", fontsize=8.5, color=GREEN)
    ax2 = ax.twinx()
    ax2.plot(yrs, d["diluted_shares"] / 1e6, color=NAVY, marker="o", lw=2.2, label="Diluted shares (M)")
    ax2.set_ylabel("Diluted shares (M)", color=NAVY)
    ax2.tick_params(axis="y", labelcolor=NAVY)
    ax2.grid(False)
    for x, v in zip(yrs, d["diluted_shares"] / 1e6):
        ax2.text(x, v + 8, f"{v:,.0f}", ha="center", fontsize=8.5, color=NAVY)
    ax.set_title("Aggressive buybacks shrinking the share count")
    ax.set_xticks(yrs)
    _footnote(fig)
    return _save(fig, cfg, "06_shares_buybacks.png")


def generate_historical_charts(cfg, H, yahoo):
    setup_style()
    paths = [
        chart_price_history(cfg, yahoo["prices"], yahoo["info"]),
        chart_revenue_growth(cfg, H),
        chart_margins(cfg, H),
        chart_fcf(cfg, H),
        chart_tpv_takerate(cfg, H),
        chart_shares_buybacks(cfg, H),
    ]
    return paths
