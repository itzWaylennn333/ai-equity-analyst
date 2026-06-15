"""Phase 4: run bull/base/bear scenarios, probability-weighted target, chart + table."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import utils, data_loader as dl, financials, wacc as wacc_mod, scenarios, charts

cfg = utils.load_config()
yahoo = dl.get_yahoo(cfg, cfg["company"]["ticker"])
facts = dl.get_edgar_facts(cfg, cfg["company"]["cik"])
H = financials.build_historical(cfg, yahoo, facts)
info = yahoo["info"]
charts.setup_style()

fy = 2025
total_debt, cash = H.loc[fy, "total_debt"], H.loc[fy, "cash"]
net_debt = total_debt - cash
base_rev = H.loc[fy, "revenue"]
shares, price, mktcap = info["sharesOutstanding"], info["currentPrice"], info["marketCap"]
W = wacc_mod.compute_wacc(cfg, mktcap, total_debt)

scen, summ = scenarios.run_scenarios(cfg, base_rev, W["wacc"], net_debt, shares, price)

disp = scen.copy()
for c in ["price_target", "dcf_gordon", "dcf_exit"]:
    disp[c] = disp[c].map(lambda v: f"${v:,.0f}")
for c in ["upside", "rev_cagr", "terminal_margin"]:
    disp[c] = disp[c].map(lambda v: f"{v*100:,.1f}%")
disp["probability"] = disp["probability"].map(lambda v: f"{v*100:.0f}%")
disp["target_fwd_pe"] = disp["target_fwd_pe"].map(lambda v: f"{v:.1f}x")
print(disp[["probability", "rev_cagr", "terminal_margin", "target_fwd_pe", "fwd_eps",
            "price_target", "upside", "dcf_gordon"]].to_string())

print("\n--- Summary ---")
print(f"Probability-weighted 12m target: ${summ['prob_weighted_target']:,.2f} "
      f"({summ['prob_weighted_upside']*100:+.1f}%)")
print(f"Bull ${summ['bull_pt']:,.0f} (+{summ['reward_to_bull']*100:.0f}%) | "
      f"Base ${summ['base_pt']:,.0f} | Bear ${summ['bear_pt']:,.0f} ({summ['risk_to_bear']*100:+.0f}%)")
print(f"Reward/risk (upside-to-bull / downside-to-bear): {summ['reward_risk_ratio']:.2f}x")

p = charts.chart_scenarios(cfg, scen, summ)
print(f"\nChart: {Path(p).name}")

# save table
out = scen.copy()
out.to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "scenarios.csv")
pd.DataFrame([summ]).to_csv(utils.resolve(cfg["output"]["tables_dir"]) / "scenario_summary.csv", index=False)
print("Saved scenarios.csv + scenario_summary.csv")
