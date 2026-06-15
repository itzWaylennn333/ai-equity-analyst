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
SRC = "Source: SEC filings & yfinance. Author analysis."


def _sym(cfg) -> str:
    return utils.currency_symbol((cfg.get("company") or {}).get("currency"))


def _name(cfg) -> str:
    c = cfg.get("company") or {}
    return c.get("name") or c.get("ticker") or "Company"


def _src_years(H) -> str:
    try:
        yrs = [int(y) for y in H.index if str(y).isdigit()]
        return f"Source: SEC filings & yfinance (FY{min(yrs)}-FY{max(yrs)}). Author analysis."
    except Exception:
        return SRC


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
    sym = _sym(cfg)
    if hi and lo:
        ax.axhspan(lo, hi, color=TEAL, alpha=0.08)
        ax.axhline(hi, color=GREY, ls="--", lw=0.8)
        ax.axhline(lo, color=GREY, ls="--", lw=0.8)
        ax.text(s.index[0], hi, f"  52-wk high {sym}{hi:,.0f}", va="bottom", fontsize=8, color=GREY)
        ax.text(s.index[0], lo, f"  52-wk low {sym}{lo:,.0f}", va="top", fontsize=8, color=GREY)
    if last:
        ax.scatter([s.index[-1]], [last], color=RED, zorder=5)
        ax.text(s.index[-1], last, f" {sym}{last:,.2f}", va="center", color=RED, fontsize=9, fontweight="bold")
    ax.set_title(f"{cfg['company'].get('ticker','')} share price (6 years)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"{sym}%d"))
    ax.set_ylabel("Share price")
    _footnote(fig, "Source: yfinance (auto-adjusted close). Author analysis.")
    return _save(fig, cfg, "01_price_history.png")


def chart_revenue_growth(cfg, H):
    sym = _sym(cfg)
    d = H.dropna(subset=["revenue"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    ax.bar(yrs, d["revenue"] / 1e9, color=NAVY, width=0.6, label=f"Revenue ({sym}B)")
    ax.set_ylabel(f"Revenue ({sym}B)")
    ax.set_title(f"{_name(cfg)}: revenue & YoY growth")
    for x, v in zip(yrs, d["revenue"] / 1e9):
        ax.text(x, v + 0.3, f"{sym}{v:,.1f}B", ha="center", fontsize=8.5, color=NAVY)
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
    _footnote(fig, _src_years(d))
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
    ax.set_title(f"{_name(cfg)}: margin trends")
    ax.set_ylabel("Margin (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(yrs)
    ax.legend(fontsize=8, ncol=2, loc="center left")
    _footnote(fig, _src_years(d))
    return _save(fig, cfg, "03_margins.png")


def chart_fcf(cfg, H):
    sym = _sym(cfg)
    d = H.dropna(subset=["fcf_reported"])
    yrs = d.index.astype(int)
    x = np.arange(len(yrs))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, d["fcf_reported"] / 1e9, width=0.4, color=NAVY, label="Reported FCF (OCF - capex)")
    ax.bar(x + 0.2, d["fcff"] / 1e9, width=0.4, color=TEAL, label="FCFF (SBC expensed, unlevered)")
    ax.set_title(f"{_name(cfg)}: free cash flow")
    ax.set_ylabel(f"{sym}B")
    ax.set_xticks(x)
    ax.set_xticklabels(yrs)
    for xi, v in zip(x - 0.2, d["fcf_reported"] / 1e9):
        ax.text(xi, v + 0.1, f"{v:,.1f}", ha="center", fontsize=8, color=NAVY)
    for xi, v in zip(x + 0.2, d["fcff"] / 1e9):
        ax.text(xi, v + 0.1, f"{v:,.1f}", ha="center", fontsize=8, color=TEAL)
    ax.legend(fontsize=8, loc="upper left")
    _footnote(fig, _src_years(d))
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
    sym = _sym(cfg)
    d = H.dropna(subset=["diluted_shares"])
    yrs = d.index.astype(int)
    fig, ax = plt.subplots()
    ax.bar(yrs, d["buybacks"] / 1e9, color=GREEN, width=0.6, label=f"Buybacks ({sym}B)")
    ax.set_ylabel(f"Share repurchases ({sym}B)", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    for x, v in zip(yrs, d["buybacks"] / 1e9):
        ax.text(x, v + 0.05, f"{sym}{v:,.1f}B", ha="center", fontsize=8.5, color=GREEN)
    ax2 = ax.twinx()
    ax2.plot(yrs, d["diluted_shares"] / 1e6, color=NAVY, marker="o", lw=2.2, label="Diluted shares (M)")
    ax2.set_ylabel("Diluted shares (M)", color=NAVY)
    ax2.tick_params(axis="y", labelcolor=NAVY)
    ax2.grid(False)
    for x, v in zip(yrs, d["diluted_shares"] / 1e6):
        ax2.text(x, v + 8, f"{v:,.0f}", ha="center", fontsize=8.5, color=NAVY)
    ax.set_title(f"{_name(cfg)}: share count & buybacks")
    ax.set_xticks(yrs)
    _footnote(fig, _src_years(d))
    return _save(fig, cfg, "06_shares_buybacks.png")


def chart_sensitivity_heatmap(cfg, grid, base_wacc, base_g, current_price):
    """grid: DataFrame index=WACC, cols=terminal g, values=intrinsic $/share."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data = grid.values.astype(float)
    vmax = np.nanmax(np.abs(data - current_price))
    norm = mcolors.TwoSlopeNorm(vmin=current_price - vmax, vcenter=current_price, vmax=current_price + vmax)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{g*100:.1f}%" for g in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{w*100:.2f}%" for w in grid.index])
    ax.set_xlabel("Terminal growth (g)")
    ax.set_ylabel("WACC")
    ax.set_title("DCF sensitivity: intrinsic value per share")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            wj = abs(grid.index[i] - base_wacc) < 1e-6 and abs(grid.columns[j] - base_g) < 1e-6
            ax.text(j, i, f"${data[i, j]:,.0f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if wj else "normal",
                    color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=1.2) if wj else None)
    ax.axhline(-0.5, color="white");
    fig.text(0.01, -0.05, f"Green = above current price ${current_price:,.2f} (undervalued); boxed = base case. "
                          "Gordon-growth TV. Author analysis.", fontsize=7.5, color=GREY)
    return _save(fig, cfg, "07_dcf_sensitivity.png")


def chart_football_field(cfg, ranges, current_price, target_price):
    """ranges: list of (label, low, high). Horizontal range bars + price/target lines."""
    sym = _sym(cfg)
    # sort each (low, high) so the bar is always drawn min->max
    ranges = [(lbl, *sorted([lo, hi])) for (lbl, lo, hi) in ranges]
    labels = [r[0] for r in ranges]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for yi, (lbl, lo, hi) in zip(y, ranges):
        ax.barh(yi, hi - lo, left=lo, height=0.5, color=TEAL, alpha=0.65, edgecolor=NAVY)
        ax.text(lo, yi, f"{sym}{lo:,.0f} ", ha="right", va="center", fontsize=8, color=GREY)
        ax.text(hi, yi, f" {sym}{hi:,.0f}", ha="left", va="center", fontsize=8, color=GREY)
    ax.axvline(current_price, color=GREY, ls="--", lw=1.4)
    ax.text(current_price, len(labels) - 0.3, f" Current {sym}{current_price:,.2f}", color=GREY, fontsize=8.5, va="bottom")
    ax.axvline(target_price, color=RED, ls="-", lw=1.8)
    ax.text(target_price, -0.7, f"Target {sym}{target_price:,.0f}", color=RED, fontsize=9, fontweight="bold", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Implied value per share ({sym})")
    ax.set_title(f"{_name(cfg)}: football field (valuation by method)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"{sym}%d"))
    _footnote(fig, "Source: author DCF & comps; 52-wk range and sell-side targets via yfinance.")
    return _save(fig, cfg, "08_football_field.png")


def chart_peer_multiples(cfg, comps_df, subj_ev_ebitda, subj_fwd_pe):
    subj = cfg["company"].get("ticker", "Subject")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4))

    def _panel(ax, metric, subj_val, title):
        core = comps_df[comps_df["group"] == "core"].dropna(subset=[metric])
        names, vals, colors = list(core["name"]), list(core[metric]), [TEAL] * len(core)
        if subj_val is not None and np.isfinite(subj_val):   # guard missing subject metric
            names, vals, colors = names + [subj], vals + [float(subj_val)], colors + [NAVY]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        if len(core):
            ax.axhline(core[metric].median(), color=GREY, ls="--", lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:,.1f}", ha="center", fontsize=7.5)

    _panel(ax1, "enterpriseToEbitda", subj_ev_ebitda, "EV / EBITDA")
    _panel(ax2, "forwardPE", subj_fwd_pe, "Forward P/E")
    fig.suptitle(f"{_name(cfg)}: valuation vs peers", fontweight="bold", fontsize=12.5)
    _footnote(fig, "Core peers only (networks/anchors excluded). Dashed = peer median. Source: yfinance. Author analysis.")
    return _save(fig, cfg, "09_peer_multiples.png")


def chart_scenarios(cfg, scen_df, summary):
    order = ["bear", "base", "bull"]
    colors = {"bear": RED, "base": NAVY, "bull": GREEN}
    price = summary["price"]
    pw = summary["prob_weighted_target"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = range(len(order))
    pts = [scen_df.loc[s, "price_target"] for s in order]
    bars = ax.bar(x, pts, color=[colors[s] for s in order], width=0.6)
    for xi, s in zip(x, order):
        pt = scen_df.loc[s, "price_target"]
        prob = scen_df.loc[s, "probability"]
        up = scen_df.loc[s, "upside"]
        ax.text(xi, pt + 1.5, f"${pt:,.0f}\n({up*100:+.0f}%)", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(xi, 2, f"{prob*100:.0f}% prob.", ha="center", fontsize=9, color="white", fontweight="bold")
        ax.text(xi, -7, f"{s.upper()}\n{scen_df.loc[s,'target_fwd_pe']:.1f}x fwd P/E", ha="center", fontsize=8.5, color=colors[s])
    ax.axhline(price, color=GREY, ls="--", lw=1.4)
    ax.text(len(order) - 0.5, price, f" current ${price:,.2f}", color=GREY, fontsize=8.5, va="bottom", ha="right")
    ax.axhline(pw, color=AMBER, ls="-", lw=2)
    ax.text(0, pw, f"prob-weighted ${pw:,.0f} ({summary['prob_weighted_upside']*100:+.0f}%)",
            color=AMBER, fontsize=9, va="bottom", fontweight="bold")
    ax.set_xticks([]);
    ax.set_ylim(-12, max(pts) * 1.2)
    ax.set_ylabel("12-month price target ($)")
    ax.set_title("Scenario price targets: skewed to the upside")
    _footnote(fig, "12m target = target fwd P/E x fwd EPS per scenario. Author analysis.")
    return _save(fig, cfg, "10_scenarios.png")


def generate_historical_charts(cfg, H, yahoo):
    setup_style()
    paths = [
        chart_price_history(cfg, yahoo["prices"], yahoo["info"]),
        chart_revenue_growth(cfg, H),
        chart_margins(cfg, H),
        chart_fcf(cfg, H),
        chart_shares_buybacks(cfg, H),
    ]
    # TPV / take-rate chart only when operating metrics exist (PayPal-style KPIs)
    if "tpv_usd_b" in H.columns and H["tpv_usd_b"].notna().any():
        paths.append(chart_tpv_takerate(cfg, H))
    return paths
