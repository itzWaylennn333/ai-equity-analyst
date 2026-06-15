# Research Foundations — Techniques & Papers

The technique choices in [ARCHITECTURE.md](ARCHITECTURE.md) are grounded in the four
research briefs below (compiled 2026-06-15). Each cites primary sources so the choices
are defensible. **Theme across all four: edges are real but small, noisy, and decaying —
build for calibrated, source-cited, honestly-bounded outputs, not hype.**

---

## A. Financial sentiment & NLP — determining *direction*

A directional engine needs 3 layers: a tone model, aspect attribution, and a signal layer
that converts tone into *direction + conviction* vs. a baseline.

1. **Domain tone models.** `FinBERT` (BERT on financial text; fast, CPU-runnable; 3-way
   polarity only) — huggingface.co/ProsusAI/finbert. `FinGPT` (LLM LoRA fine-tunes; cheap
   to retrain; graded/reasoned output; beats FinBERT) — github.com/AI4Finance-Foundation/FinGPT,
   arXiv:2306.06031. Open 7–8B Qwen-3/Llama-3 fine-tunes now match/beat FinBERT with ~5% of
   the data and handle noisy social text best — arXiv:2512.00946; survey arXiv:2507.01990.
   BloombergGPT (50B, closed) is reference-only — arXiv:2303.17564.
2. **Aspect-Based Sentiment (ABSA).** Score (aspect → polarity) pairs so "margins ↑,
   guidance ↓" aren't averaged away. `PyABSA` (arXiv:2208.01368) + LLM open-aspect
   extraction; finance framing FinXABSA arXiv:2303.02563.
3. **Tone → directional signal.** (a) tone vs. expectations/surprise — call tone dominates
   the numeric surprise over ~60 days (ScienceDirect S0378426611002901); (b) tone *change*
   vs. prior filings / 10-K YoY similarity predicts returns & crash risk ("Lazy Prices",
   Cohen-Malloy-Nguyen 2020 JF, NBER w25084); (c) text-only earnings models generate ~8%
   post-announcement drift (PEAD.txt, Phila. Fed WP21-07).
4. **Hard cases.** Loughran–McDonald financial dictionary for uncertainty/litigious/modal
   tone (sraf.nd.edu; commercial use needs a license); negation/hedging via context windows
   & transformers; route sarcastic social text to an LLM and down-weight social vs. filings.

**Recommended stack:** ingestion/segmentation → PyABSA + local-LLM aspects → FinBERT
ensembled w/ FinGPT/LoRA + SetFit conviction head → L&M lexicon overlay → signed z-scored
signal (aspect-weighted tone, Δtone, tone−consensus), validated by event-study/PEAD backtests.

**Caveats:** small/decaying/regime-dependent edges; licensing (L&M, some weights);
look-ahead leakage (pin model cutoffs, timestamp docs); social is adversarial; LLM
"conviction" rationales can be confidently wrong — keep a calibrated classifier + lexicon guardrails.

---

## B. Signal vs. noise & source credibility

Down-weight unreliable/manipulative nodes; don't treat every post equally.

1. **Bot/coordinated-inauthentic-behavior.** Botometer (now serves only frozen pre-2023
   X scores — not viable live; github.com/osome-iu/botometer-python). Graph models RGT/
   BotRGCN/**BotMoE** (arXiv:2304.06280; TwiBot-22 benchmark). Most portable: **coordination
   detection** from shared traces (cashtags/links/near-identical text/timing) → cluster dense
   subgraphs (Pacheco et al. ICWSM-2021, arXiv:2001.05658).
2. **Credibility/reputation.** *Track-record* scoring (label past calls vs. realized returns;
   strongest signal; Bar-Haim et al. EMNLP-2011, aclanthology.org/D11-1121). Network centrality
   prior (TrustRank, VLDB-2004) for cold-start. Metadata (age, follower ratio, burstiness) as
   *soft* GBM features, not hard rules. News-outlet seeds: Baly News-Media-Reliability,
   NELA-GT (free); NewsGuard/MBFC/Ad Fontes (paid).
3. **Pump-and-dump.** Market-coupled detection (joint price+volume+chatter spike; crypto F1
   ~94%, arXiv:2105.00733; equities forum 85% acc, arXiv:2301.11403); features: low float,
   cashtag piggybacking, new-account bursts, copypasta. Closest analogue: AIMM (arXiv:2512.16103).
4. **Aggregation.** `crowd-kit` Dawid–Skene / MACE (down-weight bias/spam; github.com/
   Toloka/crowd-kit); CRH/CATD truth-discovery for sparse sources; online inverse-variance
   weighting as outcomes resolve; source-dependence detection so retweets aren't double-counted.
5. **Quality filters.** fastText `lid.176` language ID; relevance gate (cashtag, ≤2 cashtags,
   drop <10-post tickers); spam/low-info heuristics (Dolma); MinHash+LSH dedup (`datasketch`);
   SemDeDup for paraphrases (arXiv:2303.09540).

**Data APIs:** Reddit (PRAW) **free** (age, karma, history); StockTwits **free** (join date,
follower counts, self-labeled bull/bear — great for track-record); X/Twitter **paid** (~$100/mo)
— the cost bottleneck. **Reddit + StockTwits are the free backbone.**

**Caveats:** organic ≠ manipulative (GME/AMC were largely organic, arXiv:2107.07361 — require
corroboration before suppressing); bot benchmarks are domain-shifted from finance; LLM bots
erode detectors (~30% drops); track-record data is sparse; guard against look-ahead.

---

## C. Predictive analytics — calibrated, honest

**Baseline reality:** equity direction is barely predictable; the strongest academic ML
studies report monthly OOS R² ~0.3–0.5% per stock, concentrated in microcaps/short-legs that
mostly vanish after costs. Build for **calibrated probabilities and modest edges**, not point
forecasts.

1. **NLP features → returns/vol.** L&M filing tone predicts filing-period abnormal returns
   (Loughran-McDonald 2011 JF); 10-K YoY textual change ("Lazy Prices", up to ~188 bps/mo
   historically, NBER w25084); earnings-call/news sentiment, topic-specific > aggregate
   (FinBERT, arXiv:2306.02136).
2. **Event-driven / PEAD.** Stocks drift with the earnings surprise for weeks (SUE long-short
   ~+18%/yr gross historically, decaying, largest in small/illiquid names; ScienceDirect
   S2214635020303750). Guidance/M&A move prices at the event but give little forward edge to outsiders.
3. **ML / time-series.** Trees + shallow nets capture nonlinear factor interactions; best
   monthly OOS R² ~0.4% (Gu, Kelly & Xiu 2020, RFS). FinBERT-LSTM improves error vs price-only
   but reports level-fit, not tradable direction (arXiv:2407.16150). Profits concentrate in
   hard-to-arbitrage names and shrink under real costs (Avramov-Cheng-Metzker 2023, Mgmt Sci) —
   treat 55–60% hit-rate claims skeptically.
4. **Composite.** Value/quality/momentum/profitability + sentiment are weakly correlated;
   z-score each cross-sectionally and blend (sentiment as a complementary sleeve, not core).
5. **Evaluation (mandatory).** Walk-forward + **purged/embargoed CV** (López de Prado);
   **Deflated Sharpe Ratio** & **Probability of Backtest Overfitting** (Bailey & López de Prado,
   SSRN 2326253) net of costs; liquidity-filter (no microcaps).

**Recommended:** cross-sectional, point-in-time pipeline → LightGBM → **calibrated probability**
(isotonic/Platt) + conformal bands; validate with purged CV; report IC, hit rate, Deflated
Sharpe, PBO net of costs. **Do NOT claim** price targets, precise forecasts, market timing, or
"beats the market." Frame as a *probabilistic directional tilt with confidence*.

---

## D. Local LLMs for agentic financial workflows

**Recommendation: local-first, hybrid-capable.** Local keeps private docs on-machine, costs
nothing per token for bulk work, and is fast; route only the hardest final reasoning to cloud
(redacted). Hybrid reportedly ~61% cheaper / ~40% lower latency than cloud-only.

1. **Models (GGUF Q4_K_M).** **Qwen2.5/Qwen3** = best all-round local family (stable tool-calling,
   clean JSON, Apache-2.0) — huggingface.co/Qwen. Llama-3.1/3.3 strong alt (3.3-70B ≈ 2023 GPT-4,
   ~48GB Q4). Phi-4 (14B) punches above weight at 8–16GB. DeepSeek leads hard math. Picks:
   8GB→Qwen2.5-7B; 16GB→Qwen2.5-14B/Phi-4; **24GB→Qwen2.5-32B (primary)**; CPU→Qwen2.5-7B (batch).
   Prefer a strong general model + RAG over a small fine-tune for analysis.
2. **Serving.** **Ollama** (easiest; GGUF; OpenAI-compatible; structured output) for dev;
   **llama.cpp** (CPU + grammars); **vLLM** (fastest GPU, batching) when concurrency grows; LM
   Studio (GUI). Quantization: Q4_K_M default, Q5 if VRAM allows; avoid <Q4 for numeric reasoning.
3. **Agents/structured output.** **PydanticAI** (typed tools + validated output) +
   **Instructor**/**Outlines** (grammar-constrained → guaranteed schema). **LangGraph** for
   multi-step/stateful report workflows. (CrewAI/AutoGen heavier alternatives.)
4. **RAG.** Structure-aware chunking (by 10-K section, 512–1,024 tok, ~20% overlap, tables
   intact); embeddings **bge-m3** or nomic-embed (domain Fin-E5 adds ~15–20%); hybrid dense+BM25
   + **bge-reranker-v2**; vector store **LanceDB** (embedded) or Qdrant (filtering/scale).
5. **Reliability.** Constrained decoding + Pydantic validation w/ retry; require chunk-ID/source
   citations; "not found" over hallucination; **deterministic Python layer owns all arithmetic.**

**Recommended (24GB):** Qwen2.5-32B on Ollama → PydanticAI agents w/ Instructor/Outlines →
LangGraph workflows → bge-m3 + LanceDB + BM25 + bge-reranker. 16GB → Qwen2.5-14B/Phi-4; CPU →
Qwen2.5-7B batch. Graduate to vLLM with scale.

---

*Caveat on sources: a few 2025–2026 arXiv items carry author-reported metrics not yet peer-reviewed,
and some pages forward-reference unreleased models — treat such claims cautiously. Commercial data/
lexicon licensing (L&M, X API, news-credibility feeds) must be verified before production use.*
