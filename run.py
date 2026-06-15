"""One-command reproduction of the PayPal equity research analysis.

    python run.py            # full pipeline: data -> analysis -> charts -> tables
    python run.py --pdf      # also render the PDF note (needs pandoc + LaTeX)

Every number in the note traces to the outputs this script regenerates from
config.yaml. Raw API pulls are cached under data/raw (offline-reproducible).
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import utils
import data_loader as dl
import financials as fin
import wacc as wacc_mod
import dcf
import comps
import scenarios
import charts


def main(render_pdf: bool = False) -> None:
    cfg = utils.load_config()
    ticker = cfg["company"]["ticker"]
    print(f"=== Equity research pipeline: {cfg['company']['name']} ({ticker}) ===")

    # 1. Data (cached) + validation
    yahoo = dl.get_yahoo(cfg, ticker)
    facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
    info = yahoo["info"]

    # 2. Historical analysis
    H = fin.build_historical(cfg, yahoo, facts)
    H.to_csv(utils.resolve(cfg["data"]["processed_dir"]) / "historical_financials.csv")
    print(f"[1/6] Historical financials: FY{int(H.index.min())}-FY{int(H.index.max())}")

    # 3. Bridge inputs
    fy = int(H.index.max())
    total_debt, cash = H.loc[fy, "total_debt"], H.loc[fy, "cash"]
    net_debt = total_debt - cash
    base_rev = H.loc[fy, "revenue"]
    shares, price, mktcap = info["sharesOutstanding"], info["currentPrice"], info["marketCap"]

    # 4. WACC + DCF
    W = wacc_mod.compute_wacc(cfg, mktcap, total_debt)
    proj = dcf.project(cfg, base_rev)
    proj.to_csv(utils.resolve(cfg["data"]["processed_dir"]) / "dcf_projection.csv")
    val = dcf.value(cfg, proj, W["wacc"], net_debt, shares, price)
    print(f"[2/6] WACC {W['wacc']*100:.2f}%  |  DCF intrinsic "
          f"${val['exit']['per_share']:.0f}-${val['gordon']['per_share']:.0f}/sh")

    # 5. Comps
    C = comps.build_comps(cfg)
    med = comps.peer_medians(C)
    print(f"[3/6] Comps: PYPL EV/EBITDA {info.get('enterpriseToEbitda'):.1f}x vs "
          f"peer median {med['ev_ebitda']:.1f}x")

    # 6. Scenarios
    scen, summ = scenarios.run_scenarios(cfg, base_rev, W["wacc"], net_debt, shares, price)
    scen.to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "scenarios.csv")
    pd.DataFrame([summ]).to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "scenario_summary.csv", index=False)
    print(f"[4/6] Scenarios: bull ${summ['bull_pt']:.0f} / base ${summ['base_pt']:.0f} / "
          f"bear ${summ['bear_pt']:.0f}  ->  prob-weighted ${summ['prob_weighted_target']:.0f} "
          f"({summ['prob_weighted_upside']*100:+.0f}%)")

    # 7. Charts
    charts.setup_style()
    charts.generate_historical_charts(cfg, H, yahoo)
    waccs = [W["wacc"] + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    gs = [0.015, 0.02, 0.025, 0.03, 0.035]
    grid = dcf.sensitivity(cfg, proj, net_debt, shares, waccs, gs)
    charts.chart_sensitivity_heatmap(cfg, grid, round(W["wacc"], 4), 0.025, price)
    pt = summ["prob_weighted_target"]
    ff = [
        ("DCF (WACC/g sensitivity)", val["exit"]["per_share"], val["gordon"]["per_share"]),
        ("Comps: EV/EBITDA (7x-10x)", (7 * info["ebitda"] - net_debt) / shares, (10 * info["ebitda"] - net_debt) / shares),
        ("Comps: Forward P/E (9x-12x)", 9 * info["forwardEps"], 12 * info["forwardEps"]),
        ("Sell-side target range", info["targetLowPrice"], info["targetHighPrice"]),
        ("52-week trading range", info["fiftyTwoWeekLow"], info["fiftyTwoWeekHigh"]),
    ]
    charts.chart_football_field(cfg, ff, price, pt)
    charts.chart_peer_multiples(cfg, C, info.get("enterpriseToEbitda"), info.get("forwardPE"))
    charts.chart_scenarios(cfg, scen, summ)
    print("[5/6] Charts regenerated -> outputs/charts/")

    # 8. Optional PDF
    if render_pdf:
        import render
        render.render_note(cfg)
        print("[6/6] PDF rendered -> outputs/research_note.pdf")
    else:
        print("[6/6] Skipped PDF (use --pdf to render).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="also render the PDF note")
    args = ap.parse_args()
    main(render_pdf=args.pdf)
