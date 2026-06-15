# Equity Research Note — PayPal Holdings, Inc. (PYPL)

A reproducible, model-backed equity research note on PayPal. A config-driven Python
engine pulls financial data (yfinance + SEC EDGAR), runs a DCF and comparable-company
analysis, performs scenario and sensitivity analysis, and generates every chart and
table used in the written note. The note is the product; the engine makes every
number in it defensible.

> **Deliverable:** [`outputs/research_note.pdf`](outputs/research_note.pdf) — a ~5–6 page
> initiating-coverage note. Data as of 2026-06-15 (FY2025 10-K).

## Thesis TL;DR — **BUY, $59 12-month target (+42%)**
The market prices PayPal as a melting ice cube; we see a **mispriced cash machine**.
At **7.2× forward earnings and a ~15% FCF yield**, the market is discounting structural
free-cash-flow decline — yet under CEO Alex Chriss the profit engine is *inflecting*
(operating margin 13.9% → 18.3%, ROIC ~9% → ~17%), and ~$6B/yr of buybacks have
compounded EPS from $2.09 to $5.41. The take-rate erosion (2.02% → 1.85%) is real but
substantially priced. **We don't need PayPal to win the checkout war — only not to lose
it outright**, which is all the price requires. Probability-weighted target $59 with a
**6.8× reward-to-risk** skew (bull $81 / base $59 / bear $36).

| | |
|---|---|
| **Ticker** | PYPL (NASDAQ) |
| **Rating** | **BUY** |
| **12-month price target** | **$59** (+42%) |
| **Current price** | $41.53 |
| **Valuation** | DCF intrinsic $80–$94; ~6× EV/EBITDA, 7.2× fwd P/E |

## How to reproduce (clean checkout)
```bash
# 1. Create/activate a Python 3.13 environment, then:
python -m pip install -r requirements.txt

# 2. (optional) verify data sources are reachable:
python tests/test_connectivity.py

# 3. Run the full pipeline (data -> financials -> wacc -> dcf -> comps -> scenarios -> charts):
python run.py

# 4. Also render the PDF note (needs pandoc + a LaTeX engine):
python run.py --pdf
```
**Render toolchain:** `winget install JohnMacFarlane.Pandoc`, plus a LaTeX engine —
[tectonic](https://tectonic-typesetting.github.io/) (self-contained, recommended) or
MiKTeX/TeX Live. `src/render.py` auto-detects pandoc + the engine (incl. per-user installs).

Cached raw pulls live in `data/raw/` (committed, so the analysis reproduces offline),
with provenance logged in `data/raw/_manifest.json`.

## Platform: analyze *any* ticker

This repo is also a generalized, config-driven analysis platform. PayPal ships locked,
hand-approved assumptions (`companies/PYPL.yaml`); any other ticker auto-derives a
sensible base case from its own history, which you then tune.

```bash
# CLI — value any ticker (assumptions auto-derived; banks/insurers/REITs skip the DCF)
python run.py --ticker NKE
python run.py --ticker MSFT

# Interactive interface — tunable sliders -> live football field, sensitivity heatmap,
# scenarios; upload documents; export. (Streamlit + Plotly)
streamlit run app.py
```

- **Add a company:** copy `companies/_TEMPLATE.yaml` to `companies/<TICKER>.yaml`
  (minimum = the `company:` block; everything else auto-derives). CIK auto-resolves
  for US tickers.
- **Integrity guardrails:** GAAP operating income from EDGAR with a yfinance fallback;
  EDGAR net-income cross-validation; terminal-growth cap; WACC>g guard; sector router
  that declines a FCFF DCF for banks/insurers/REITs/pre-revenue biotech; currency-aware.
- **Document ingestion:** `src/ingest.py` parses uploaded PDF/HTML/DOCX/XLSX/CSV/TXT into
  classified, chunked, provenance-tagged records (feeds RAG/extraction in later phases).
- **Design & roadmap:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (6-layer system,
  interface, local-LLM stack); technique research + citations in
  [`docs/RESEARCH.md`](docs/RESEARCH.md). Built so far: P1 generalize, P2 ingestion,
  P3 interface. Next: P4 sentiment → P5 credibility → P6 RAG+agents → P7 predictive → P8 synthesis.

_Use an isolated environment (the project uses a conda env `equity-analyzer`)._

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
