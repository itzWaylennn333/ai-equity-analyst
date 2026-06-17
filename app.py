"""Equity Intelligence Platform — interactive interface (Streamlit).

    streamlit run app.py

Pick a ticker (or upload documents), then tune valuation variables in the sidebar;
the visualizations and KPIs update live. PayPal uses its locked assumptions; any other
ticker auto-derives a sensible base case you then tune. Numbers are deterministic and
cached, so slider moves are near-instant.
"""
from __future__ import annotations

import sys
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import registry
import data_loader as dl
import financials as fin
import assumptions as asm
import wacc as wacc_mod
import dcf
import comps as comps_mod
import scenarios as scen_mod
import charts as charts_mod
import ingest
import rag
import sentiment as sentiment_mod

st.set_page_config(page_title="Equity Intelligence Platform", layout="wide")

NAVY, TEAL, RED, GREEN, AMBER, GREY = "#003087", "#009cde", "#c0392b", "#1e8449", "#d68910", "#6b7280"


# --------------------------------------------------------------------------- #
# Data loading (cached per ticker)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Pulling data & building historicals...")
def load_ticker(ticker: str) -> dict:
    cfg = registry.load_company(ticker)
    comp = cfg["company"]
    cik = comp.get("cik") or dl.resolve_cik(cfg, ticker)
    yahoo = dl.get_yahoo(cfg, ticker)
    info = yahoo["info"]
    if not comp.get("name") or comp["name"] == ticker:
        comp["name"] = info.get("longName") or ticker
    if not registry.has_company_file(ticker):
        comp["sector_template"] = registry.detect_sector_template(info)
    facts = dl.get_edgar_facts(cfg, cik)
    H = fin.build_historical(cfg, yahoo, facts)

    comp["currency"] = info.get("financialCurrency") or info.get("currency") or comp.get("currency", "USD")
    fy = int(H.index.max())
    total_debt = H.loc[fy, "total_debt"] if pd.notna(H.loc[fy, "total_debt"]) else (info.get("totalDebt") or 0)
    cash = H.loc[fy, "cash"] if pd.notna(H.loc[fy, "cash"]) else (info.get("totalCash") or 0)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    mktcap = info.get("marketCap")
    shares = info.get("sharesOutstanding")
    if not shares and mktcap and price:
        shares = mktcap / price
    if not mktcap and shares and price:
        mktcap = shares * price
    if not (price and shares and mktcap):
        raise ValueError("yfinance did not return price / shares / market cap for this ticker.")
    C = comps_mod.build_comps(cfg) if cfg["peers"].get("core") else pd.DataFrame()
    med = comps_mod.peer_medians(C) if len(C) else {"ev_ebitda": np.nan, "forward_pe": np.nan}
    asm.ensure_assumptions(cfg, H, info, med.get("ev_ebitda"))
    W = wacc_mod.compute_wacc(cfg, mktcap, total_debt)

    for d in (cfg["output"]["charts_dir"], cfg["output"]["tables_dir"], cfg["data"]["processed_dir"]):
        Path(charts_mod.utils.resolve(d)).mkdir(parents=True, exist_ok=True)
    charts_mod.setup_style()
    charts_mod.generate_historical_charts(cfg, H, yahoo)

    tmpl = cfg["sector_templates"].get(comp.get("sector_template", "standard"), {})
    return {
        "cfg": cfg, "H": H, "info": info, "base_year": fy,
        "net_debt": total_debt - cash, "base_rev": H.loc[fy, "revenue"],
        "shares": shares, "price": price, "mktcap": mktcap, "currency": comp["currency"],
        "wacc_base": W["wacc"], "wacc_detail": W, "comps": C, "peer_ev_ebitda": med.get("ev_ebitda"),
        "supported": tmpl.get("method") not in ("unsupported", "ffo_nav"),
        "sector_note": tmpl.get("note", ""),
    }


# --------------------------------------------------------------------------- #
# Sidebar: ticker + documents + tunable controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Equity Intelligence")
ticker = st.sidebar.text_input("Ticker", value="PYPL").strip().upper()
go_btn = st.sidebar.button("Load / refresh", type="primary")

uploads = st.sidebar.file_uploader("Upload documents (10-K, transcripts, models...)",
                                   accept_multiple_files=True,
                                   type=["pdf", "html", "htm", "docx", "xlsx", "xls", "csv", "txt"])

if go_btn or "data" not in st.session_state or st.session_state.get("ticker") != ticker:
    try:
        st.session_state["data"] = load_ticker(ticker)
        st.session_state["ticker"] = ticker
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load {ticker}: {e}")
        st.stop()

D = st.session_state["data"]
cfg0, H, info = D["cfg"], D["H"], D["info"]
price, shares, net_debt, base_rev = D["price"], D["shares"], D["net_debt"], D["base_rev"]

st.sidebar.markdown("### Tune assumptions")
proj0 = cfg0["valuation"]["projections"]
g_avg0 = float(np.mean(proj0["revenue_growth"]))
m_avg0 = float(np.mean(proj0["ebit_margin"]))
wacc = st.sidebar.slider("WACC (%)", 4.0, 16.0, round(D["wacc_base"] * 100, 2), 0.1) / 100
gterm = st.sidebar.slider("Terminal growth (%)", 0.0, float(cfg0["valuation"]["terminal"]["growth_cap"] * 100),
                          round(cfg0["valuation"]["terminal"]["growth"] * 100, 2), 0.1) / 100
exitm = st.sidebar.slider("Exit EV/EBITDA (x)", 4.0, 20.0, float(cfg0["valuation"]["terminal"]["exit_ev_ebitda"]), 0.5)
g_avg = st.sidebar.slider("Revenue growth, avg (%)", -5.0, 25.0, round(g_avg0 * 100, 1), 0.5) / 100
m_avg = st.sidebar.slider("Operating margin (%)", 1.0, 50.0, round(m_avg0 * 100, 1), 0.5) / 100
st.sidebar.markdown("**Scenario probabilities**")
p_bull = st.sidebar.slider("Bull %", 0, 100, int(round(cfg0["scenarios"]["bull"]["probability"] * 100)), 5)
p_bear = st.sidebar.slider("Bear %", 0, 100, int(round(cfg0["scenarios"]["bear"]["probability"] * 100)), 5)
p_base = max(0, 100 - p_bull - p_bear)
st.sidebar.caption(f"Base % = {p_base}")

# Build a tuned config copy and recompute (deterministic, instant)
cfg = copy.deepcopy(cfg0)
horizon = cfg["valuation"]["horizon_years"]
cfg["valuation"]["terminal"]["growth"] = gterm
cfg["valuation"]["terminal"]["exit_ev_ebitda"] = exitm
cfg["valuation"]["projections"]["revenue_growth"] = [round(g_avg, 4)] * horizon
cfg["valuation"]["projections"]["ebit_margin"] = [round(m_avg, 4)] * horizon
cfg["scenarios"]["bull"]["probability"] = p_bull / 100
cfg["scenarios"]["base"]["probability"] = p_base / 100
cfg["scenarios"]["bear"]["probability"] = p_bear / 100

proj = dcf.project(cfg, base_rev, base_year=D["base_year"])
val = dcf.value(cfg, proj, wacc, net_debt, shares, price)
scen, summ = scen_mod.run_scenarios(cfg, base_rev, wacc, net_debt, shares, price)


# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title(f"{cfg['company'].get('name', ticker)} ({ticker})")
if not D["supported"]:
    st.warning(f"Sector guardrail: **{cfg['company']['sector_template']}** — {D['sector_note']} "
               "DCF shown for reference only; treat with caution.")
k = st.columns(5)
k[0].metric("Price", f"${price:,.2f}")
k[1].metric("DCF intrinsic (Gordon)", f"${val['gordon']['per_share']:,.0f}", f"{val['gordon']['upside']*100:+.0f}%")
k[2].metric("Prob-weighted target", f"${summ['prob_weighted_target']:,.0f}", f"{summ['prob_weighted_upside']*100:+.0f}%")
k[3].metric("WACC", f"{wacc*100:.2f}%")
k[4].metric("Reward / risk", f"{summ['reward_risk_ratio']:.1f}x")

tabs = st.tabs(["Summary", "Valuation", "Scenarios", "Historical", "Comps", "Documents",
                "Sentiment", "Predictive", "Report"])

# --- Summary ---
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Valuation bridge")
        st.write(f"- **Enterprise value (Gordon):** ${val['gordon']['ev']/1e9:,.1f}B")
        st.write(f"- **Equity value:** ${val['gordon']['equity']/1e9:,.1f}B  |  **Net debt:** ${net_debt/1e9:,.1f}B")
        st.write(f"- **Intrinsic / share:** ${val['gordon']['per_share']:,.2f} (Gordon) / "
                 f"${val['exit']['per_share']:,.2f} (exit {exitm:.1f}x)")
        st.write(f"- **PV of explicit FCFF:** ${val['pv_fcff']/1e9:,.1f}B  |  "
                 f"**Terminal % of EV:** {val['gordon']['terminal_pct_of_ev']*100:.0f}%")
    with c2:
        st.subheader("Snapshot")
        for lbl, key, f in [("Market cap", "marketCap", 1e9), ("EV", "enterpriseValue", 1e9),
                            ("Fwd P/E", "forwardPE", 1), ("EV/EBITDA", "enterpriseToEbitda", 1),
                            ("Beta", "beta", 1)]:
            v = info.get(key)
            st.write(f"**{lbl}:** " + (f"{v/f:,.1f}" + ("B" if f > 1 else "") if isinstance(v, (int, float)) else "n/a"))

# --- Valuation: football field + sensitivity ---
with tabs[1]:
    ff = [("DCF (exit → Gordon)", val["exit"]["per_share"], val["gordon"]["per_share"])]
    if info.get("ebitda"):
        ff.append(("Comps EV/EBITDA 7-10x", (7*info["ebitda"]-net_debt)/shares, (10*info["ebitda"]-net_debt)/shares))
    if info.get("forwardEps"):
        ff.append(("Comps fwd P/E 9-12x", 9*info["forwardEps"], 12*info["forwardEps"]))
    if info.get("targetLowPrice") and info.get("targetHighPrice"):
        ff.append(("Sell-side range", info["targetLowPrice"], info["targetHighPrice"]))
    if info.get("fiftyTwoWeekLow") and info.get("fiftyTwoWeekHigh"):
        ff.append(("52-week range", info["fiftyTwoWeekLow"], info["fiftyTwoWeekHigh"]))
    fig = go.Figure()
    for lbl, lo, hi in ff:
        lo, hi = sorted([lo, hi])
        fig.add_trace(go.Bar(y=[lbl], x=[hi - lo], base=[lo], orientation="h",
                             marker_color=TEAL, hovertemplate=f"{lbl}: ${lo:,.0f}-${hi:,.0f}<extra></extra>"))
    fig.add_vline(x=price, line_dash="dash", line_color=GREY, annotation_text=f"Price ${price:,.0f}")
    fig.add_vline(x=summ["prob_weighted_target"], line_color=RED,
                  annotation_text=f"Target ${summ['prob_weighted_target']:,.0f}")
    fig.update_layout(title="Football field", showlegend=False, height=360, xaxis_title="$ / share")
    st.plotly_chart(fig, width="stretch")

    waccs = [wacc + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    gs = [gterm + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
    grid = dcf.sensitivity(cfg, proj, net_debt, shares, waccs, gs)
    hm = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{g*100:.1f}%" for g in grid.columns], y=[f"{w*100:.2f}%" for w in grid.index],
        text=[[f"${v:,.0f}" for v in row] for row in grid.values], texttemplate="%{text}",
        colorscale="RdYlGn", zmid=price))
    hm.update_layout(title="DCF sensitivity: $/share (rows WACC, cols terminal g)",
                     height=360, xaxis_title="terminal growth", yaxis_title="WACC")
    st.plotly_chart(hm, width="stretch")
    st.dataframe(proj.assign(**{c: proj[c].round(0) for c in ["revenue", "ebit", "fcff", "ebitda"]})
                 [["revenue", "rev_growth", "ebit_margin", "fcff", "ebitda"]])

# --- Scenarios ---
with tabs[2]:
    colors = {"bull": GREEN, "base": NAVY, "bear": RED}
    order = ["bear", "base", "bull"]
    fig = go.Figure()
    for s in order:
        pt = scen.loc[s, "price_target"]
        fig.add_trace(go.Bar(x=[s.upper()], y=[pt], marker_color=colors[s],
                             text=[f"${pt:,.0f}<br>{scen.loc[s,'upside']*100:+.0f}%"], textposition="outside"))
    fig.add_hline(y=price, line_dash="dash", line_color=GREY, annotation_text=f"price ${price:,.0f}")
    fig.add_hline(y=summ["prob_weighted_target"], line_color=AMBER,
                  annotation_text=f"prob-wtd ${summ['prob_weighted_target']:,.0f}")
    fig.update_layout(title="Scenario price targets", showlegend=False, height=420, yaxis_title="$ / share")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(scen[["probability", "target_fwd_pe", "fwd_eps", "price_target", "upside", "dcf_gordon"]])

# --- Historical (static charts) ---
with tabs[3]:
    cdir = Path(charts_mod.utils.resolve(cfg["output"]["charts_dir"]))
    imgs = sorted(cdir.glob("0*.png"))
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        cols[i % 2].image(str(p), width="stretch")

# --- Comps ---
with tabs[4]:
    C = D["comps"]
    if len(C):
        st.dataframe(C[["name", "group", "enterpriseToRevenue", "enterpriseToEbitda", "trailingPE", "forwardPE"]])
    else:
        st.info("No peer set configured for this ticker. Add peers in companies/<TICKER>.yaml.")

# --- Documents ---
with tabs[5]:
    st.subheader("Document ingestion")
    if uploads:
        tmpdir = Path(charts_mod.utils.resolve(cfg["data"]["processed_dir"])) / ticker / "_uploads"
        tmpdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for uf in uploads:
            fp = tmpdir / uf.name
            fp.write_bytes(uf.getbuffer())
            paths.append(fp)
        res = ingest.ingest_files(cfg, ticker, paths)
        st.dataframe(pd.DataFrame(res))
        st.caption(f"Stored {sum(r.get('n_chunks',0) for r in res)} chunks for RAG/extraction (P6).")
    chunks = ingest.load_chunks(cfg, ticker)
    if chunks:
        st.write(f"**{len(chunks)} chunks** in the document store.")
        st.json(chunks[0])

        st.markdown("---")
        st.markdown("**Ask the filing — grounded RAG (P6)**")
        st.caption("Hybrid dense+BM25 retrieval → LLM rerank → cite-or-abstain extraction: every "
                   "figure is verified verbatim against a cited chunk and support-checked, or the "
                   "system abstains. No fabricated numbers.")
        if st.button("Build / refresh index", key="rag_build"):
            with st.spinner("Embedding chunks (bge-m3) + building LanceDB index…"):
                info = rag.build_index(cfg, ticker, chunks)
            st.success(f"Indexed {info['n_chunks']} chunks (dim {info['dim']}).")
        q = st.text_input("Question", placeholder="What were net revenues in fiscal 2025?", key="rag_q")
        if q:
            try:
                with st.spinner("Retrieving + extracting…"):
                    out = rag.extract(cfg, ticker, q)
                r = out["retrieval"]
                st.caption(f"retrieval: top cosine {r['top_similarity']} · "
                           f"{'reranked' if r.get('reranked') else 'no rerank'} · "
                           f"{'confident' if r['confident'] else 'low confidence'}")
                if out["found"]:
                    a, cite = out["answer"], out["citation"]
                    st.success(f"**{a['value']}**" + (f" {a['unit']}" if a.get("unit") else "")
                               + (f"  ({a['period']})" if a.get("period") else ""))
                    st.markdown(f"> {cite['quote']}")
                    st.caption(f"📎 {cite['source_file']} / {cite['section']} · chunk "
                               f"`{cite['chunk_id']}` · verified verbatim + support-checked")
                else:
                    st.warning(f"No grounded answer — **abstained**. {out['reason']}")
            except Exception as e:  # noqa: BLE001
                st.error(f"RAG failed: {e}. Click **Build / refresh index** first; "
                         "ensure Ollama is serving bge-m3 + the configured LLM.")

        st.markdown("**Reconcile filing vs. computed (P6.3)**")
        st.caption("Cross-checks the latest-FY headline figures extracted from the filing against the "
                   "deterministic EDGAR/yfinance numbers and flags conflicts. Build the index first.")
        if st.button("Reconcile latest fiscal year", key="rag_recon"):
            try:
                import reconcile
                with st.spinner("Extracting + cross-checking…"):
                    rc = reconcile.reconcile_financials(cfg, ticker)
                if rc.get("error"):
                    st.warning(rc["error"])
                else:
                    st.write(("✅ All figures reconcile." if rc["ok"] else "⚠️ Conflicts found.")
                             + f"  {rc['counts']}")
                    st.dataframe(pd.DataFrame([{"metric": r["metric"], "FY": r["period"],
                                 "deterministic": r["expected"], "filing": r["filing_value"],
                                 "status": r["status"], "rel_diff": r["rel_diff"]} for r in rc["items"]]))
            except Exception as e:  # noqa: BLE001
                st.error(f"Reconcile failed: {e}. Build the index + ensure Ollama is up.")
    else:
        st.info("Upload documents above to populate the per-ticker document store.")

# --- Sentiment (P4) ---
with tabs[6]:
    st.subheader("Sentiment (directional)")
    method = cfg["sentiment"].get("method", "lexicon")
    chunks = ingest.load_chunks(cfg, ticker)
    if not chunks:
        st.info("Upload documents (Documents tab) to run sentiment. Engine method = "
                f"**{method}** (config `sentiment.method`): `lexicon` (default, no deps), `finbert` "
                "(needs transformers), or `llm` (Ollama endpoint, e.g. your DGX Spark).")
    else:
        try:
            rep = sentiment_mod.analyze(cfg, ticker, chunks, method=method)
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall tone", f"{rep['overall_tone']:+.2f}", rep["direction"])
            c2.metric("Conviction", f"{rep['conviction']:.0%}")
            c3.metric("Uncertainty rate", f"{rep['uncertainty_rate']*100:.2f}%")
            if rep["by_aspect"]:
                asp = pd.DataFrame([{"aspect": a, "tone": v["tone"], "mentions": v["mentions"]}
                                    for a, v in rep["by_aspect"].items()]).sort_values("tone")
                fig = go.Figure(go.Bar(x=asp["tone"], y=asp["aspect"], orientation="h",
                                       marker_color=[GREEN if t >= 0 else RED for t in asp["tone"]],
                                       text=[f"{t:+.2f} ({m})" for t, m in zip(asp["tone"], asp["mentions"])]))
                fig.update_layout(title=f"Aspect sentiment ({method})", height=340,
                                  xaxis_title="tone (-1 bearish .. +1 bullish)")
                st.plotly_chart(fig, width="stretch")
            st.caption("Filings skew negative (risk-factor language); read the **aspect breakdown** and "
                       "tone-vs-prior-filing change as the directional signal, not the absolute level. "
                       "Lexicon tone is coarse — enable `finbert`/`llm` for nuance.")
            with st.expander("Most positive / negative excerpts"):
                for e in rep.get("top_positive", []):
                    st.markdown(f"🟢 **({e['tone']:+.2f})** {e['text']}")
                for e in rep.get("top_negative", []):
                    st.markdown(f"🔴 **({e['tone']:+.2f})** {e['text']}")
            cred = rep.get("credibility")
            if cred and not cred.get("error"):
                st.markdown("---")
                st.markdown("**Credibility & noise (Layer 2b)**")
                q, risk = cred["quality"], cred["manipulation_risk"]
                k1, k2, k3 = st.columns(3)
                k1.metric("Credibility-weighted tone", f"{cred['weighted_tone']:+.2f}", cred["direction"])
                k2.metric("Nodes kept", f"{q['kept']}/{cred['n_nodes']}", f"-{q['dropped']} filtered",
                          delta_color="off")
                k3.metric("Manipulation risk", risk["level"].upper())
                st.caption("Trust weight = quality (language · relevance · info-density · dedup) × source "
                           "credibility. Flag: " + "; ".join(risk["reasons"][:2]))
                if cred["excluded"]:
                    with st.expander(f"Excluded nodes ({len(cred['excluded'])}) · signal drivers"):
                        st.write("**Filtered out:**"); st.dataframe(pd.DataFrame(cred["excluded"]))
                        if cred["top_weighted"]:
                            st.write("**Top-weighted (driving the signal):**")
                            st.dataframe(pd.DataFrame(cred["top_weighted"]))
                if not cred["social"]["available"]:
                    st.caption("ℹ️ Account track-record / bot / coordination layers need a social source "
                               "(StockTwits/Reddit, with accounts + timestamps) — not wired yet.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Sentiment ({method}) failed: {e}. Try method=lexicon, or check the LLM endpoint.")
with tabs[7]:
    st.info("**Predictive layer (roadmap P7):** factor + sentiment composite → calibrated probability of "
            "positive forward excess return, validated with purged CV + Deflated Sharpe. See docs/ARCHITECTURE.md §7.")

# --- Report ---
with tabs[8]:
    st.subheader("Export")
    st.write("Generate a PDF research note from the **current tuned state** "
             "(requires pandoc + a LaTeX engine; PayPal ships a full hand-written note).")
    if ticker == "PYPL" and Path(charts_mod.utils.resolve(cfg["output"]["pdf_path"])).exists():
        with open(charts_mod.utils.resolve(cfg["output"]["pdf_path"]), "rb") as f:
            st.download_button("Download PYPL research note (PDF)", f, file_name="PYPL_research_note.pdf")
    st.caption("Auto-generated multi-section notes for arbitrary tickers arrive with the synthesis layer (P8).")
