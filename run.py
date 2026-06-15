"""One-command equity analysis for ANY ticker.

    python run.py                 # default ticker (PYPL)
    python run.py --ticker NKE    # analyze any ticker (assumptions auto-derived)
    python run.py --ticker PYPL --pdf

Pipeline: registry/config -> data (cached) -> historical analysis -> auto-derive any
missing assumptions -> WACC -> DCF -> comps -> scenarios -> charts (-> optional PDF).
PayPal uses its locked, hand-approved assumptions; other tickers auto-derive from history.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import utils
import registry
import data_loader as dl
import financials as fin
import assumptions as asm
import wacc as wacc_mod
import dcf
import comps
import scenarios
import charts


def main(ticker: str = "PYPL", render_pdf: bool = False) -> dict:
    cfg = registry.load_company(ticker)
    comp = cfg["company"]
    print(f"=== Equity analysis: {comp.get('name', ticker)} ({ticker}) ===")

    # Ensure output/processed directories exist (per-ticker namespacing)
    for d in (cfg["output"]["charts_dir"], cfg["output"]["tables_dir"], cfg["data"]["processed_dir"]):
        utils.resolve(d).mkdir(parents=True, exist_ok=True)

    # CIK (resolve for US tickers if not specified) + data
    cik = comp.get("cik") or dl.resolve_cik(cfg, ticker)
    yahoo = dl.get_yahoo(cfg, ticker)
    info = yahoo["info"]
    if not comp.get("name") or comp["name"] == ticker:
        comp["name"] = info.get("longName") or ticker

    # Auto-detect sector template for UNCONFIGURED tickers (configured files win).
    if not registry.has_company_file(ticker):
        comp["sector_template"] = registry.detect_sector_template(info)
    tmpl = cfg["sector_templates"].get(comp.get("sector_template", "standard"), {})
    supported = tmpl.get("method") not in ("unsupported", "ffo_nav")  # FCFF DCF applies?

    facts = dl.get_edgar_facts(cfg, cik)
    print(f"[1/6] Data: {comp['name']} | CIK {cik or 'n/a'} | sector '{comp['sector_template']}' "
          f"| price ${info.get('currentPrice')}")

    # Historical analysis (always produced)
    H = fin.build_historical(cfg, yahoo, facts)
    H.to_csv(utils.resolve(cfg["data"]["processed_dir"]) / f"historical_{ticker}.csv")
    print(f"[2/6] Historical: FY{int(H.index.min())}-FY{int(H.index.max())}")
    charts.setup_style()
    charts.generate_historical_charts(cfg, H, yahoo)

    # Integrity guardrail: do NOT force a FCFF DCF on a business it doesn't fit.
    if not supported:
        print(f"[GUARDRAIL] sector '{comp['sector_template']}' -> {tmpl.get('note')}")
        print("            Skipping DCF/scenarios; produced historical analysis + charts only.")
        return {"cfg": cfg, "supported": False, "historical": H}

    # Bridge inputs
    fy = int(H.index.max())
    total_debt = H.loc[fy, "total_debt"] if pd.notna(H.loc[fy, "total_debt"]) else (info.get("totalDebt") or 0)
    cash = H.loc[fy, "cash"] if pd.notna(H.loc[fy, "cash"]) else (info.get("totalCash") or 0)
    net_debt = total_debt - cash
    base_rev = H.loc[fy, "revenue"]
    shares, price, mktcap = info["sharesOutstanding"], info["currentPrice"], info["marketCap"]

    # Comps (peer median used for exit multiple; tolerate empty peer set)
    C = comps.build_comps(cfg) if cfg["peers"].get("core") else pd.DataFrame()
    med = comps.peer_medians(C) if len(C) else {"ev_ebitda": np.nan, "forward_pe": np.nan}
    peer_ev_ebitda = med.get("ev_ebitda")

    # Fill any missing assumptions from history (no-op for PYPL)
    asm.ensure_assumptions(cfg, H, info, peer_ev_ebitda)

    # WACC + DCF
    W = wacc_mod.compute_wacc(cfg, mktcap, total_debt)
    proj = dcf.project(cfg, base_rev)
    proj.to_csv(utils.resolve(cfg["data"]["processed_dir"]) / f"dcf_projection_{ticker}.csv")
    val = dcf.value(cfg, proj, W["wacc"], net_debt, shares, price)
    print(f"[3/6] WACC {W['wacc']*100:.2f}%  |  DCF intrinsic "
          f"${val['exit']['per_share']:.0f}-${val['gordon']['per_share']:.0f}/sh")
    if len(C):
        print(f"[4/6] Comps: EV/EBITDA {info.get('enterpriseToEbitda')} vs peer median {peer_ev_ebitda:.1f}x")
    else:
        print("[4/6] Comps: no peer set configured (skipped peer median)")

    # Scenarios
    scen, summ = scenarios.run_scenarios(cfg, base_rev, W["wacc"], net_debt, shares, price)
    scen.to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "scenarios.csv")
    print(f"[5/6] Scenarios: bull ${summ['bull_pt']:.0f} / base ${summ['base_pt']:.0f} / "
          f"bear ${summ['bear_pt']:.0f}  ->  prob-weighted ${summ['prob_weighted_target']:.0f} "
          f"({summ['prob_weighted_upside']*100:+.0f}%)")

    # Valuation charts (historical charts already generated above)
    waccs = [W["wacc"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    gs = [cfg["valuation"]["terminal"]["growth"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    grid = dcf.sensitivity(cfg, proj, net_debt, shares, waccs, gs)
    charts.chart_sensitivity_heatmap(cfg, grid, round(W["wacc"], 4),
                                     round(cfg["valuation"]["terminal"]["growth"], 4), price)
    # Football field (only rows with available data)
    ff = [("DCF (WACC/g sensitivity)", val["exit"]["per_share"], val["gordon"]["per_share"])]
    if info.get("ebitda"):
        ff.append(("Comps: EV/EBITDA (7x-10x)", (7*info["ebitda"]-net_debt)/shares, (10*info["ebitda"]-net_debt)/shares))
    if info.get("forwardEps"):
        ff.append(("Comps: Forward P/E (9x-12x)", 9*info["forwardEps"], 12*info["forwardEps"]))
    if info.get("targetLowPrice") and info.get("targetHighPrice"):
        ff.append(("Sell-side target range", info["targetLowPrice"], info["targetHighPrice"]))
    if info.get("fiftyTwoWeekLow") and info.get("fiftyTwoWeekHigh"):
        ff.append(("52-week trading range", info["fiftyTwoWeekLow"], info["fiftyTwoWeekHigh"]))
    charts.chart_football_field(cfg, ff, price, summ["prob_weighted_target"])
    if len(C):
        charts.chart_peer_multiples(cfg, C, info.get("enterpriseToEbitda"), info.get("forwardPE"))
    charts.chart_scenarios(cfg, scen, summ)
    print("[6/6] Charts -> " + cfg["output"]["charts_dir"])

    if render_pdf:
        import render
        render.render_note(cfg)
        print("      PDF -> " + cfg["output"]["pdf_path"])

    return {"cfg": cfg, "wacc": W, "valuation": val, "scenarios": (scen, summ)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="PYPL")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    main(ticker=args.ticker.upper(), render_pdf=args.pdf)
