# Project Brief — Portfolio-Grade Equity Research Note

> Saved verbatim from the kickoff brief for reference. The note is the product;
> the engine is what makes the note trustworthy.

## 1. Role and intent
An experienced equity research analyst + strong Python engineer, helping the analyst
(final-year engineering student) build a portfolio-grade
equity research note. Target: finance analyst roles (equity research, IBD, markets)
for 2027 summer recruiting. The artifact must prove the ability to think like an
analyst: form a view on a company, value it properly, and defend it. It is a
**research deliverable** backed by a clean, reproducible model — not a software
showcase. Optimise every tradeoff for "would a first-year ER associate or an IBD
interviewer respect this and find it defensible."

## 2. What we are building
A reproducible, model-backed equity research note on a single public company:
1. **The deliverable** — a 4–6 page PDF research note: rating (Buy/Hold/Sell),
   model-derived price target, thesis, business & industry analysis, competitive
   positioning, valuation (DCF + comps), catalysts, honest risk/bear case.
2. **The engine** — a config-driven Python valuation model that pulls financial
   data, runs a DCF and comparable-companies analysis, performs scenario and
   sensitivity analysis, and generates every chart and table the note uses.

## 3. Why this matters
Two audiences: the recruiter who skims, and the interviewer who interrogates.
The interviewer is the binding constraint — every claim, assumption, and number
must be defensible live.
- **I own the view.** Core judgment calls (company, thesis, key assumptions, final
  rating) are surfaced as explicit decisions, never silently chosen.
- **Integrity is non-negotiable.** No fabricated numbers. Every figure traces to a
  cited source or an explicit stated assumption. Missing data is flagged, not invented.
- **The bear case must be real.** Steelman the opposing view.
- **Assumptions must be benchmarked.** Terminal growth ≤ long-run nominal GDP
  (~3–4%). Margins/growth justified vs the company's own history and peers.

## 4. How to work
- Plan first; work in phases with checkpoints; ask the kickoff questions first.
- Flag every key judgment call with a proposed default + reasoning.
- Improve on the brief where warranted.
- Cache aggressively (save raw API pulls to disk).
- Validate data (cross-check revenue, net debt, share count vs the annual report).

## 5. Constraints
- **Stack:** Python (pandas, numpy, matplotlib). Config in YAML.
- **Data (free):** yfinance (prices, financials, market cap, beta, shares, peers);
  SEC EDGAR (`https://data.sec.gov`) for 10-K/10-Q exact figures + MD&A. Risk-free
  rate = live 10Y government bond yield (US 10Y for USD valuation). ERP = current
  published figure (Damodaran standard), stated in the assumptions log. Flag gaps
  for non-US names.
- **Qualitative:** industry/competitive narrative from data + filings; mark claims
  needing a source; use web where available.
- **PDF:** markdown → clean PDF. Prefer pandoc + LaTeX; fall back to weasyprint.
  Charts embedded as vector or high-res PNG.

## 6. Kickoff questions (answered)
1. Company/ticker — **PayPal (PYPL)**
2. Market — **US-listed**
3. Thesis lean — **Let the analysis settle it**
4. Projection horizon — **5 explicit years + terminal value**
Plus setup: **pandoc+LaTeX** rendering, **git with a commit per phase**.

## 7. Company selection criteria
Understandable business; well-covered with clean data (large/mid-cap, several years
of filings); not structurally hard to value (avoid banks, insurers, financials;
avoid pre-revenue/single-asset biotech); has a real bull/bear debate; stable enough
to model (no transformative M&A/restructuring). — PYPL qualifies: payments processor
(DCF applies cleanly), large-cap, clean multi-year filings, genuine debate over
branded-checkout take-rate erosion.

## 8. Architecture (refined in the plan)
Config-driven so all assumptions live in `config.yaml` and the valuation is fully
reproducible. Refinements added vs the original tree: `run.py` one-command
orchestrator, `data/raw/_manifest.json` provenance log, `tests/` integrity checks
(incl. terminal-growth cap guard), `src/utils.py` shared helpers.

## 9. Phasing & checkpoints (◆ = stop for the analyst)
- **P0 Setup & plan** — questions, scaffold, config, requirements, BRIEF. ◆ pick company + confirm plan.
- **P1 Data & historical analysis** — pull/cache/validate; build historical picture + charts. ◆ review summary + gaps.
- **P2 Thesis formation** — articulate thesis (2–4 args) + steelmanned bear case. ◆ lock thesis + rating direction.
- **P3 Valuation** — WACC, DCF, comps, sensitivity, football field. ◆ approve key assumptions.
- **P4 Scenarios & charts** — bull/base/bear + probability-weighted target; finalize charts/tables.
- **P5 Write the note** — full draft (§11 structure). ◆ review + own prose and call.
- **P6 Render & document** — PDF, README, assumptions log; reproduce from clean checkout.

## 10. Valuation methodology
- **FCFF** = EBIT×(1−tax) + D&A − CapEx − ΔNWC
- **WACC** = (E/V)·Re + (D/V)·Rd·(1−tax); Re = Rf + β·ERP (CAPM);
  Rd = interest expense / total debt (or Rf + spread); E = mkt cap, D = total debt,
  V = E+D; Rf = live 10Y; ERP = Damodaran; β = levered beta.
- **Terminal value** — compute BOTH: Gordon growth `FCFF_n(1+g)/(WACC−g)` with
  g ≤ ~3–4%; exit multiple `EBITDA_n × peer exit EV/EBITDA`.
- **EV → equity → per share:** EV = ΣPV(FCFF) + PV(TV); Equity = EV − Net Debt
  (Net Debt = Total Debt − Cash); Intrinsic/share = Equity / diluted shares;
  Upside = Intrinsic/Price − 1.
- **Sensitivity:** 2-D table/heatmap of price across WACC × terminal-growth.
- **Comps:** 5–8 named peers; EV/Revenue, EV/EBITDA, EV/EBIT, P/E, PEG; apply peer
  median (+ range) to target metrics; justify each peer.
- **Football field:** reconcile DCF range, comps (EV/EBITDA), comps (P/E), 52-wk
  range, sell-side targets if available; mark current price + target.
- **Scenarios:** bull/base/bear, each an explicit coherent assumption set; combine
  into a probability-weighted target with implied risk/reward.

### Minimum chart set
1. Share price history + 52-wk range; 2. Revenue & growth (hist+proj); 3. Margin
trends; 4. FCF history + projection; 5. DCF sensitivity heatmap; 6. Football field;
7. Bull/base/bear targets; 8. Peer multiples comparison.

## 11. Written note — structure
Page-1 summary (rating, target, thesis bullets, key stats) · Investment thesis ·
Business overview · Industry & market · Competitive positioning · Financial analysis
· Valuation (DCF + assumptions + sensitivity + comps + football field + scenarios) ·
Catalysts · Risks & bear case · Conclusion. Target 4–6 dense pages; tight analytical
prose; charts/tables do heavy lifting.

## 12. Definition of done
4–6 page professional PDF; model-derived rating + target on page 1; DCF reproducible
from config with benchmarked assumptions; named-peer comps + football field;
bull/base/bear with probability-weighted target; genuine (non-strawman) bear case;
every figure sourced or stated; `assumptions.md` logs every input + justification;
`README.md` explains thesis + reproduction; clean modular config-driven cached code;
the analyst has reviewed and can defend thesis, assumptions, and final call.
