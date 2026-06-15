# Assumptions Log — PayPal (PYPL) Valuation

Every key input, its value, and its justification. All live in `config.yaml` (single
source of truth); the valuation reproduces from there. Data as of 2026-06-15.
Rating: **BUY** · 12-month target **$59** (+42%) · current **$41.53**.

## Market & company inputs (observed)
| Input | Value | Source |
|---|---|---|
| Current price | $41.53 | yfinance |
| Shares outstanding | 882.1M | yfinance (most current) |
| Market cap | $36.6B | yfinance |
| Total debt | $9.99B | EDGAR FY25 10-K (= yfinance) |
| Cash & equivalents | $8.05B | EDGAR FY25 10-K (= yfinance) |
| Net debt (strict) | $1.94B | Total debt − cash; **excludes** pass-through customer funds & loans receivable |
| FY2025 net revenue | $33.17B | EDGAR / yfinance (tie out) |
| FY2025 GAAP operating income | $6.07B | **EDGAR** (yfinance's $6.40B reclassification rejected) |

## WACC (8.57%)
| Input | Value | Justification |
|---|---|---|
| Risk-free rate | 4.49% | **Live** US 10Y Treasury (`^TNX`), pulled at runtime |
| Beta (levered) | 1.30 | yfinance 1.34; regressions 1.22 (2y) / 1.43 (5y) weekly vs S&P 500 → central 1.30 |
| Equity risk premium | 4.23% | Damodaran implied ERP, S&P 500, Jan-2026 vintage (cited) |
| Cost of equity (CAPM) | 9.99% | Rf + β·ERP |
| Pre-tax cost of debt | 4.42% | FY25 interest expense ÷ total debt; consistent with IG (A-/A3) profile |
| Marginal tax (for debt) | 24% | US federal 21% + state |
| Weights | 78.6% E / 21.4% D | Market value of equity; book≈market debt |

## DCF base-case drivers (FY2026E–FY2030E)
| Driver | Value | Benchmark / rationale |
|---|---|---|
| Revenue growth | 4.5%, 4.5%, 4.0%, 3.8%, 3.5% (~4.1% CAGR) | Below FY25's +4.3%; bakes in continued take-rate drag, partly offset by volume + new monetization |
| Operating (EBIT) margin | 18.3% → 18.7% | ~Flat vs FY25 GAAP 18.3%; modest efficiency, mostly reinvested in checkout defense |
| D&A % revenue | 2.9% | FY25 actual |
| CapEx % revenue | 2.6% | FY25 actual (asset-light) |
| ΔNWC % revenue | 0.5% | Modest core working-capital drag; ex pass-through funds/loan dynamics |
| Tax rate (NOPAT) | 23% | Normalized; FY25's 16.8% was tax-credit flattered |
| SBC treatment | **Expensed** | Embedded in EBIT, **not** added back to FCFF (conservative) |
| Terminal growth | 2.5% | Below the 4.0% hard cap (long-run nominal GDP); mature payments franchise |
| Exit EV/EBITDA (cross-check TV) | 9.0× | Modest re-rate from current ~6×; discount to ~12.6× peer median |

→ **DCF intrinsic value: $80 (exit) – $94 (Gordon).** Sensitivity (WACC 7.6–9.6% × g 1.5–3.5%) spans **$73–$137 — entirely above the current price.**

## Comparable companies
Core peers (median drivers): **Block (XYZ), Global Payments (GPN), Adyen (ADYEN.AS),
Fidelity National (FIS)**. Networks **Visa (V), Mastercard (MA)** shown for context only,
excluded from the median (different toll-road economics). **Fiserv (FI) intended but
data unavailable from source at pull time — disclosed gap.** Core medians: EV/EBITDA
12.6×, forward P/E 9.8×. PYPL trades at ~6× / 7.2× — a discount to the median (though
GPN/FIS screen lower on forward P/E).

## Scenarios & probability-weighted target
| Scenario | Prob. | Rev CAGR | Terminal margin | Target fwd P/E | Fwd EPS | 12m PT |
|---|---|---|---|---|---|---|
| Bull | 25% | 6.4% | 21.5% | 13.5× | $6.00 | $81 (+95%) |
| Base | 50% | 4.1% | 18.7% | 10.2× | $5.77 | $59 (+42%) |
| Bear | 25% | 1.2% | 17.0% | 6.5× | $5.50 | $36 (−14%) |

**Probability-weighted target: $58.61 (+41%). Reward/risk: 6.8×.** Forward EPS per
scenario from yfinance consensus ($5.77) flexed modestly; the swing factor is the
**target multiple** (re-rating thesis), not near-term EPS.

## Key judgment calls (interview-ready)
1. **Intrinsic value ($80–94) ≠ price target ($59).** The DCF shows the downside is
   protected and the market prices terminal decline; the 12-month target reflects a
   *partial* re-rating (~10× fwd P/E), not full convergence.
2. **SBC expensed, not added back** — conservative for a tech/payments name.
3. **GAAP operating income from EDGAR**, not the data vendor (which misstated it).
4. **Net debt excludes customer funds & the loan book** — they are pass-through /
   operating items, not corporate capital.
5. **Take-rate compression is real and sourced** — the thesis is that stabilization
   (not reversal) plus buybacks is enough, given the price.
