# Equity Research Note — PayPal Holdings, Inc. (PYPL)

A reproducible, model-backed equity research note on PayPal. A config-driven Python
engine pulls financial data (yfinance + SEC EDGAR), runs a DCF and comparable-company
analysis, performs scenario and sensitivity analysis, and generates every chart and
table used in the written note. The note is the product; the engine makes every
number in it defensible.

> **Status:** Phase 0 complete (setup & scaffold). Rating, price target, and thesis
> are **not yet determined** — they are produced by the model and locked at the
> Phase 2–3 checkpoints. Nothing here is a recommendation yet.

## Thesis TL;DR
_To be written at Phase 2 (thesis) / Phase 5 (note). The central debate: is PayPal's
branded-checkout take-rate erosion **structural** (Apple Pay / Shop Pay taking the
high-margin core) or **cyclical-and-fixable** (new management re-monetizing a
400M+ account franchise that generates large free cash flow)?_

| | |
|---|---|
| **Ticker** | PYPL (NASDAQ) |
| **Rating** | _TBD — Phase 2_ |
| **Price target** | _TBD — Phase 3_ |
| **Current price** | see latest data pull |

## How to reproduce (clean checkout)
```bash
# 1. Create/activate a Python 3.13 environment, then:
python -m pip install -r requirements.txt

# 2. Verify data sources are reachable (optional smoke test):
python tests/test_connectivity.py

# 3. Run the full pipeline (data -> financials -> wacc -> dcf -> comps -> charts):
python run.py            # (added incrementally across phases)

# 4. Render the PDF note (requires pandoc + a LaTeX distribution; falls back to weasyprint):
#    handled by the render step in Phase 6
```
Cached raw pulls live in `data/raw/` (committed, so the analysis reproduces offline),
with provenance logged in `data/raw/_manifest.json`.

## Project structure
```
equity-research-PYPL/
├── README.md  BRIEF.md  config.yaml  requirements.txt
├── run.py                    # one-command end-to-end reproduce (added across phases)
├── data/
│   ├── raw/                  # cached API pulls + _manifest.json (provenance)
│   └── processed/            # cleaned data the model consumes
├── src/
│   ├── utils.py              # config loader, caching, provenance manifest
│   ├── data_loader.py        # yfinance / EDGAR pulls + caching + validation
│   ├── financials.py         # historical growth, margins, ratios, FCF
│   ├── wacc.py               # CAPM cost of equity, cost of debt, WACC
│   ├── dcf.py                # projections, FCFF, terminal value, PV, sensitivity
│   ├── comps.py              # peer multiples + implied valuation
│   ├── scenarios.py          # bull / base / bear, probability-weighted target
│   └── charts.py             # all chart generation
├── tests/                    # integrity & sanity checks (incl. terminal-growth cap)
├── notebooks/analysis.ipynb  # exploratory sanity checks
├── outputs/
│   ├── charts/  tables/      # generated figures & tables
│   └── research_note.pdf     # THE DELIVERABLE
└── note/
    ├── research_note.md       # the written note (source)
    └── assumptions.md         # every key assumption, value, and justification
```

## Data sources
- **yfinance** — prices, historical financials, market cap, beta, shares, peer data.
- **SEC EDGAR** (`https://data.sec.gov`) — exact 10-K/10-Q figures + MD&A; PayPal
  CIK `0001633917`. Key figures (revenue, net debt, share count) are validated
  against the filings, not taken from the API alone.
- **Risk-free rate** — current US 10Y Treasury yield, pulled live.
- **Equity risk premium** — current published Damodaran figure (cited in `assumptions.md`).

## Reproducibility & integrity
- All assumptions live in `config.yaml` (single source of truth).
- Every figure in the note traces to a cited source or an explicit stated assumption.
- Terminal growth is hard-capped at long-run nominal GDP; enforced in `tests/`.

_Environment: Anaconda Python 3.13.9 at `C:\Users\wayle\anaconda3\python.exe` (Windows 11)._
