# Phase 1 — Historical Analysis & Data Findings (PYPL)

_Data as of pull on 2026-06-15. Last actuals: FY2025 (10-K filed 2026-02-03). Latest interim: Q1 2026 (10-Q filed 2026-05-05)._

## 1. Data validation (EDGAR filings vs yfinance)
Key figures were cross-checked against SEC EDGAR XBRL (ground truth). **The core
figures tie out exactly**, so we can trust the yfinance backbone:

| FY2025 figure | EDGAR (filing) | yfinance | Verdict |
|---|---|---|---|
| Net income | $5,233M | $5,233M | ✅ exact |
| Diluted shares | 968M | 968M | ✅ exact |
| Cash & equivalents | $8,049M | $8,049M | ✅ exact |
| Long-term debt | $9,987M | $9,987M | ✅ exact |
| Operating income (GAAP) | **$6,065M** | $6,396M | ⚠️ yfinance reclassifies — **we use the filing** |

The operating-income discrepancy is the one trap: yfinance's "Operating Income" row
does **not** match the filing. The filed figure ($33,172M revenue − $27,107M total
opex = $6,065M) reconciles exactly with EDGAR `OperatingIncomeLoss`. The model takes
GAAP operating income from EDGAR, not yfinance.

## 2. Historical summary (FY2022–FY2025)
Full table in [outputs/tables/historical_summary.md](../outputs/tables/historical_summary.md).
Charts in [outputs/charts/](../outputs/charts/). Headlines:

- **Revenue decelerating:** $27.5B → $33.2B, growth +8.2% → +6.8% → **+4.3%**.
- **But profitability improving:** GAAP operating margin 13.9% → **18.3%**; net margin
  8.8% → 15.8%; ROIC ~9% → **~16.7%**. Operating leverage under CEO Alex Chriss
  (since Sept-2023): opex discipline, SBC cut from 5.0% to **3.0%** of revenue.
- **EPS compounding far faster than revenue:** $2.09 → **$5.41**, because diluted
  shares fell 1,158M → **968M** on ~$6B/yr buybacks (now ~882M shares outstanding).
- **Cash generation is the bull's anchor:** reported FCF ~$5.6B (FY25). On the
  stricter, SBC-expensed, unlevered basis we model, FCFF ≈ **$4.1B**.
- **Balance sheet ~net cash:** strict net debt ~$1.9B; **net cash** (−$0.4B) once
  short-term investments are included. (Customer funds & loans receivable are
  excluded as pass-through / operating items — a key modeling judgment.)
- **Valuation is depressed:** $41.53, **7.8× trailing P/E**, ~15% FCF yield, near the
  52-week low ($38–$80 range), down ~85% from the 2021 peak.

## 3. The central tension, quantified (sourced from 10-Ks)
| FY | TPV | Take rate | Transactions | Active accts |
|---|---|---|---|---|
| 2022 | $1.36T | 2.02% | — | — |
| 2023 | $1.53T | 1.95% | — | — |
| 2024 | $1.68T | 1.89% | 26.3B | 434M |
| 2025 | $1.79T | 1.85% | 25.4B (−4%) | 439M (+1%) |

Volume compounds double-digit while **take rate has fallen ~17bps in three years**.
The FY2025 10-K states the mechanism verbatim: transaction revenue rose only +2%
vs TPV +7% *"due to … favorable changes in merchant mix to lower cost merchants
within our Braintree products."* Unbranded (Braintree) volume is diluting blended
economics; branded checkout (the high-margin core) is the contested battleground.
Note transactions **fell 4%** in FY25 even as TPV rose — volume is shifting to fewer,
larger-ticket unbranded payments, and engagement (txns/account 60.6 → 57.7) softened.

## 4. Data gaps & how they're handled
| Gap | Status / plan |
|---|---|
| TPV, take rate, transactions, active accounts (not in yfinance/XBRL) | **Closed** — hand-sourced from 10-K MD&A with citations in `data/processed/operating_metrics.csv`. |
| Branded vs unbranded revenue/volume split, transaction-margin-dollar bridge | PayPal discloses these qualitatively + some figures in MD&A; **to be extracted at Phase 2** for the thesis. Branded/unbranded TPV split is not cleanly disclosed every year — will flag any number that is estimated. |
| Income-statement detail pre-FY2022 (yfinance gives only 4 yrs) | EDGAR has operating income/net income back to ~2013 if a longer trend is wanted; current charts use FY2022–25 + 6-yr price history. |
| Beta (yfinance = 1.336) | Cross-check via regression vs S&P 500 in Phase 3. |
| Q1 2026 run-rate (10-Q filed 2026-05-05) | Cached; to be incorporated for the current-year bridge in Phase 3. |
| Peer financials (Block, Fiserv, GPN, Adyen, FIS) | Deferred to Phase 3 (comps). |

## 5. What the data implies for the thesis (for Phase 2 debate)
- **Bull seed:** a ~$36B-market-cap business throwing off ~$5–6B FCF (≈15% yield),
  net cash, buying back ~6–7% of shares a year, with margins and ROIC *rising* — at
  7.8× earnings. The market is pricing terminal decline.
- **Bear seed:** the high-margin branded core is structurally losing checkout share;
  take rate is in steady, sourced decline; transactions are now *falling*; growth is
  carried by low-margin unbranded volume. "Cheap" can stay cheap if transaction
  margin dollars stop growing.
- The whole call hinges on **transaction-margin-dollar growth** (≈+5–7%/yr lately):
  does it hold, re-accelerate (branded monetization, Venmo, agentic commerce), or
  fade? That is the Phase 2 decision.
