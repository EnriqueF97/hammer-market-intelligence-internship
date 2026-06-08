# Project Logbook — Modeling News-Driven Liquidity Dynamics in Commodities Markets

**Student:** Enrique Favila Martínez | **Institution:** Radboud University MSc AI | **Host:** Hammer Market Intelligence | **Supervisor:** Dr. Lejla Batina | **Last updated:** May 2026

---

## Phase 0 — Commodity Selection

### Started with sugar, pivoted to WTI

- Original goal: model news-driven liquidity in sugar markets
- Data sources evaluated: TradingMap, UN Comtrade, FAOSTAT — all too aggregated (monthly/annual), no intraday data
- NLP test on agriculture news: FinBERT, SEC-BERT, ChatGPT, DeepSeek all disagreed significantly on same articles → model choice is non-trivial methodological decision

### Why sugar was dropped

- Strong seasonal patterns dominate any news signal
- Limited geopolitical sensitivity
- No liquid hourly futures data available

### Pivot to WTI Crude Oil

- Active 24h market, abundant public data
- Reacts sharply to OPEC, geopolitics, macro releases
- yfinance provides 2yr hourly OHLCV for free

---

## Phase 1 — Data Infrastructure + EIA Baseline

### Research questions

- **RQ1:** Do bearish news events have larger/more persistent liquidity impact than bullish?
- **RQ2:** What is the lag structure — how quickly does the market absorb news?
- **RQ3 (optional):** Cross-commodity spillovers

### Data sources selected

- **yfinance `CL=F`:** 11,219 hourly OHLCV records, 2yr coverage (Mar 2024 – Mar 2026)
- **EIA API `WCRSTUS1`:** 322 weekly inventory reports, classified bearish/bullish
- **GDELT:** Free-text news headlines + URLs, downloaded week by week

### Liquidity variables derived from OHLCV

- `log_volume` — log(Volume+1), normalizes skewed distribution
- `price_range` — High−Low, Parkinson (1980) volatility proxy
- `log_return` — log(Close*t/Close*{t-1})
- `amihud` — |log_return|/Volume, Amihud (2002) illiquidity ratio

### EIA OLS Baseline — first evidence of asymmetry

- Event windows ±4h around EIA publication (101 events × 993 records)
- `is_bearish` coef = 0.210, p = 0.030 ✅
- Shock magnitude not significant (p = 0.782) — direction matters, not size
- R² = 0.024 — motivates NLP component
- Price range peaks at hour −1 → market anticipates EIA before publication

---

## Phase 2 — NLP Pipeline

### GDELT download

- Query: "crude oil WTI price" → throttling issues → fix: browser User-Agent header + 8s sleep between requests
- GDELT cap: 250 articles/request — high-activity weeks truncated (acceptable limitation)
- Later expanded to 8 queries → 51,948 raw articles, deduplicated to 16,326 English

### Article body scraping

- GDELT only provides title + URL, no body text
- Built BeautifulSoup scraper: removes nav/footer/scripts, extracts first 3 paragraphs (>50 chars)
- ~80% success rate — failures from paywalls, Cloudflare blocks, JS-rendered pages
- **Key pipeline decision:** save raw CSV immediately after scraping before any filtering → never need to re-scrape

### Body quality filtering — evolved from blacklist to allow-list

- Early approach: blacklist of known bad phrases (Cloudflare, error pages, cookie notices)
- Final approach: allow-list requiring 400+ chars, energy/financial keywords, no press release language
- Result: 7,756 valid bodies out of 16,326 articles (47.5%)

### Temporal alignment — causality fix

- Original: `dt.round('h')` → could assign article to hour BEFORE publication (causality violation)
- Fix: `dt.ceil('h')` → always rounds forward to next complete trading hour
- **Impact:** coefficients nearly doubled after fix (0.114 → 0.243)
- Additional fix: articles published outside trading hours forward-assigned to next market open instead of discarded → recovered ~2,374 articles
- Added `assignment_gap` column: hours between publication and assigned trading hour

### FinBERT sentiment scoring

- Model: ProsusAI/finbert, MPS acceleration on M1
- Two scores per article: `title_sentiment` (headline only) and `full_sentiment` (title + body, fallback to title if no valid body)
- 7,756 articles got title+body input; 5,934 got title-only fallback

### Headline bias — novel methodological finding

- **43.6% divergence** between title-only and title+body sentiment among valid body articles (n=7,755)
- χ² = 2050.15, p < 0.001 — replicated on 6x larger dataset, result is robust
- Neutral headlines mask bearish content **39.6%** of the time
- Positive → negative divergence: **29.9%** of positive headlines have bearish bodies
- Title+body confidence higher (0.863 vs 0.805) — body provides genuine disambiguation
- **Methodological conclusion:** title+body selected as primary FinBERT input

---

## Phase 3 — Lag Analysis + VAR Attempt

### Contemporaneous OLS — first significant GDELT results

- With 13,690 articles and ceil alignment: both coefficients now significant
- `is_bearish` coef = 0.133, p < 0.001 ✅
- `is_bullish` coef = 0.103, p = 0.004 ✅
- Bearish > Bullish — first evidence of asymmetry at contemporaneous hour

### Lag OLS — peak at lag+6, asymmetry in timing

| Lag  | Bearish       | Bullish         |
| ---- | ------------- | --------------- |
| +1h  | ✅ sig        | ✅ sig          |
| +2h  | ❌            | ❌              |
| +3h  | ❌            | ✅ sig          |
| +4h  | ✅ sig        | ✅ sig          |
| +6h  | ✅ sig (peak) | ✅ sig (peak)   |
| +8h  | ❌            | ❌              |
| +12h | ❌            | ✅ sig negative |

- **RQ2 preliminary answer:** news impact peaks at lag+6 hours
- **RQ1 preliminary answer:** bearish > bullish at all significant lags; bullish has delayed negative reversal at lag+12 (mean reversion); different timescales
- Scheduled news (EIA) → pre-announcement effect at hour −1; unscheduled news (GDELT) → contemporaneous + lag effects dominate

### VAR model — built but abandoned

**Setup:**

- 4 variables: log_volume, sentiment_score, log_return, price_range
- All stationary (ADF p < 0.0001)
- Optimal lag: 24 (all criteria agree)
- 10,825 hourly observations

**Problem — signal sparsity:**

- 50% of hourly rows have `sentiment_score = 0` (no articles that hour)
- IRF confidence bands straddle zero throughout — no statistically significant impulse response
- Only significant sentiment coefficient: L8.sentiment_score → log_volume (p=0.011), consistent with lag OLS

**Fixes attempted:**

- Expanded to 8 GDELT queries → coverage improved from 14% to 53% of trading hours with articles
- Filtered to contemporaneous articles only (gap < 2h) → VAR results nearly identical
- Root cause: hourly aggregation with zeros is fundamentally the wrong structure for this data

**Conclusion:** VAR abandoned. Lag OLS results are cleaner and already answer both RQs.

---

## Current State — April 2026

### What's done ✅

- Full data pipeline: yfinance → EIA → GDELT → scraping → alignment → FinBERT
- Headline bias experiment complete (novel contribution)
- Lag OLS complete with significant results
- VAR attempted and documented

### Key numbers

- 13,690 articles aligned with price data
- 7,756 with valid body text
- Peak news impact: lag+6 hours
- Headline bias divergence: 41.6% (valid body articles)

### Where to resume — next step

- **Event study:** for each article, measure log_volume in ±12h window, average bearish vs bullish curves separately → clean visual answer to RQ1
- **Optional:** local projections (Jordà 2005) as VAR replacement — more robust, produces IRF-like curves, widely cited in macro-finance
- Lag OLS already answers RQ2 — event study needed to formally answer RQ1

---

## Open Questions + Limitations

- Bid-ask spread unavailable from yfinance — using volume + Amihud as proxies
- Original proposal mentioned Hammer proprietary data — using public sources instead (needs discussion with Dr. Batina)
- ARIMA in original proposal superseded by lag OLS + event study approach

---

## Phase 4 — Planned (Next)

### Core architecture shift

- Moving from lag OLS + abandoned VAR → **Temporal Fusion Transformer (TFT)**
- Attention weights replace regression coefficients as evidence for RQ1 and RQ2
- Richer input features replace FinBERT binary labels

### Why TFT over lag OLS for RQ1 and RQ2

- **RQ2:** lag OLS gives 7 discrete coefficient points — TFT gives continuous attention distribution over all hours simultaneously
- **RQ1:** lag OLS has one coefficient per sentiment direction — TFT captures nonlinear interactions (e.g. bearish + high VIX + strong dollar = disproportionate volume spike)
- Attention weights are direct interpretable evidence, not inferred from coefficients

### LLM feature extraction — replacing FinBERT labels

- GPT-4o-mini or Claude Haiku API (~$1-2 for all 13,690 articles)
- Per article, extract structured JSON:
  - `sentiment_score` — continuous −1 to +1 (vs FinBERT's 3-class label)
  - `magnitude` — event importance 0 to 1
  - `event_type` — geopolitical / supply / demand / macro / inventory / technical
  - `entities` — OPEC, Russia, Fed, Iran, Saudi Arabia, etc.
  - `certainty` — 0 to 1 (speculative vs confirmed)
  - `price_direction` — bearish / bullish / neutral
  - `time_horizon` — immediate / short_term / long_term
- New notebook: `09_llm_features.ipynb`
- Prompt design is critical — test on 10 articles before full batch

### Additional parallel market variables

- **DXY (US Dollar Index)** — WTI priced in USD, strong inverse correlation, available via yfinance
- **VIX** — market volatility / risk-off signal, available via yfinance
- **Cushing Oklahoma inventory** — physical WTI delivery hub, more granular than national EIA, available via EIA API
- **OPEC+ events** — meeting dates and production decisions, structured event flags
- **Seasonality** — hour of day, day of week, month — computed from timestamp, no external source needed
- New notebook: `10_parallel_features.ipynb`

### Why these parallel variables matter

- DXY and VIX are what professional WTI traders monitor in real time
- Same bearish news has different impact depending on macro context — TFT can learn this
- Without these, the model sees sentiment in a vacuum; with them it sees sentiment in context

### Execution plan

- **Week 1-2:** LLM feature extraction (09)
- **Week 3:** Download + align parallel market variables (10)
- **Week 4:** Event study for RQ1 — fast, clean, complements lag OLS
- **Week 5-6:** TFT implementation with pytorch-forecasting
- **Week 7:** Analysis — attention weights, asymmetry, TFT vs lag OLS comparison
- **Week 8:** Thesis writing

---

## Results of phase 4 so far (11 may)

## Phase 4 — TFT Implementation and Results

### Architecture and training

- Replaced VAR with Temporal Fusion Transformer (TFT) via pytorch-forecasting 1.7.0
- Training done on Google Colab T4 GPU
- Model: hidden_size=32, attention_head_size=4, dropout=0.1, 112k parameters
- val_loss = 0.204 (best at epoch 21, early stopped)
- Checkpoint saved locally at `01_data/models/tft_wti.ckpt`

### Input features

- **LLM features (Claude Haiku):** sentiment_score, magnitude, event_type, certainty, price_direction, time_horizon
- **Market context:** log_volume, price_range, log_return, amihud, DXY, VIX
- **Temporal covariates:** hour, day_of_week, month, is_wednesday, is_us_session
- 10,797 hourly rows, 48h encoder window, 1h prediction horizon
- Train/val split: 80/20, ~8,637 training hours, ~2,160 validation hours

### RQ2 — Lag structure (attention weights)

- Peak attention at **-4h** — the model pays most attention to conditions 4 hours ago
- Consistent with lag OLS peak at lag+6 — both identify the 4-6 hour window as most informative
- Two distinct attention clusters identified:
  - **Short-term:** -2h to -5h — immediate news absorption window
  - **Daily memory:** -27h to -28h — same time yesterday, captures daily market cycles
- TFT adds insight over lag OLS: daily memory effect was invisible in the 12h OLS window

### RQ1 — Asymmetry (feature importance + directional analysis)

- **sentiment_score importance = 53%** — single most important feature by large margin
- Confirms LLM continuous scoring outperforms FinBERT binary labels
- Top features: sentiment_score (0.53) > log_volume (0.09) > DXY (0.08) > log_return (0.06)
- VIX importance = 0.005 — negligible, DXY already captures risk signal for WTI
- **Directional asymmetry:**
  - Bearish predicted volume: 8.775
  - Bullish predicted volume: 8.700
  - Bearish > Bullish confirmed, consistent with lag OLS and EIA baseline
  - T-test p=0.56 — not statistically significant (underpowered: only 156/144 validation hours)
- **Asymmetry conclusion:** direction consistent across all models but TFT underpowered for significance test; lag OLS (p<0.001) is the stronger evidence for RQ1

### Key methodological findings

- LLM feature extraction (Haiku) produces dramatically richer sentiment signal than FinBERT
- event_type ranks above magnitude — category of news more informative than size
- Neutral news hours have highest predicted volume (8.875) — uncertainty drives trading
- TFT best used for RQ2 (attention weights) and feature importance; lag OLS better for RQ1 significance

### Where to resume

- Write up combined RQ1 and RQ2 answers using all evidence:
  - RQ1: EIA baseline (p=0.030) + lag OLS (p<0.001) + TFT feature importance (53%) + TFT directional (bearish > bullish)
  - RQ2: lag OLS peak at +6h + TFT attention peak at -4h + daily memory at -27/-28h
- Optional: run event study for cleaner RQ1 visual
- Thesis writing — logbook has all material condensed

### Reflect on keeping more data (up to may 2026) or to keep it pre war (feb 2026)

We are keeping the current TFT model with pre-war news, current val loss = 0.209 which is good. and the lag peak is at 4h, meaning that traders have higher volume after 4 hours the news was published.

---

## Phase 5 — Data Expansion and Pipeline Migration to DB (In Progress)

### Migration from CSV to SQLite database

- All data now lives in `01_data/wti_thesis.db` — CSVs kept as read-only backups
- DB tables: `articles`, `liquidity`, `llm_features`, `market_context`, `eia_events`, `opec_events`
- All notebooks updated to read/write from DB instead of CSV

### EIA events — structured into DB

- Created `eia_events` table with: `date`, `inventory_mbbl`, `inventory_change`, `news_direction`, `datetime_et`
- Key fix: `datetime_et` stored as UTC (`15:00+00:00`) after converting from US/Eastern and ceiling to next whole hour
- 331 weekly records from 2020-01-03 to 2026-05-01

### EIA alignment with market_context

- Added `eia_surprise` and `is_eia_release` columns to `market_context`
- Used `pd.merge_asof` with `direction='backward'` — each trading hour gets the most recent EIA value before it
- `eia_surprise` holds the weekly inventory change, constant across the week until next Wednesday release
- `is_eia_release = 1` only at the exact hour of publication (Wednesday ~15:00 UTC)
- Verified: value changes exactly once per EIA week at the correct timestamp

### Market context expansion

- Re-ran notebook 01 to extend `market_context` to May 2026
- DXY coverage: 99.9%, VIX coverage: 99.9% (16h forward fill for off-hours)
- `market_context` now covers May 2024 → May 2026

### GDELT news expansion — new queries added

- Added 5 high-signal geopolitical queries to existing 8:
  - `Iran sanctions oil`
  - `Saudi Arabia oil production`
  - `China oil demand`
  - `Russia oil exports`
  - `oil supply disruption`
- Downloaded January 2026 → May 2026 gap
- Raw CSV appended (never replaced) — 51,948 → growing
- Clean CSV updated to 22,795 English deduplicated articles
- DB `articles` table updated to 22,795 articles
- **4,221 new articles** from March 2026 to May 2026
- **8,918 articles pending body scraping** (new articles + some from expanded queries)

### Body scraping — resumed and near-complete (14 May 2026)

- Notebook 04 rewritten to use DB exclusively: reads `body IS NULL` articles, `UPDATE` per row, commit every 50
- Resume-capable: re-running skips already-scraped rows
- Scraper ran in two sessions; 5,970 articles still pending after first pass, reduced to 2,749 after second
- **Current articles breakdown (22,795 total):**
  - valid body (`body_valid=1`): **13,550**
  - invalid body (scraped, failed quality filter): 6,496
  - still pending (`body IS NULL`): **2,749**
  - never attempted: 0

---

## Phase 5 — DB Schema and Pipeline State (14 May 2026)

### Current DB schema (`01_data/wti_thesis.db`)

| Table            | Rows   | Columns                                                                                                                                                                                                                            |
| ---------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `articles`       | 22,795 | `id`, `title`, `datetime`, `domain`, `url`, `body`, `body_valid`                                                                                                                                                                   |
| `market_context` | 11,232 | `datetime_hour`, `log_volume`, `price_range`, `log_return`, `amihud`, `dxy`, `vix`, `cushing_inventory`, `eia_surprise`, `is_eia_release`                                                                                          |
| `liquidity`      | 22,795 | `article_id`, `datetime_hour`, `assignment_gap`, `log_volume`, `price_range`, `log_return`, `amihud`, `sentiment_score`, `magnitude`, `event_type`, `entities`, `certainty`, `price_direction`, `time_horizon`, `has_llm_features` |
| `llm_features`   | 12,024 | `article_id`, `sentiment_score`, `magnitude`, `event_type`, `entities`, `certainty`, `price_direction`, `time_horizon`                                                                                                             |
| `eia_events`     | 331    | `date`, `inventory_mbbl`, `inventory_change`, `news_direction`, `datetime_et`                                                                                                                                                      |
| `opec_events`    | 0      | `id`, `datetime`, `event_type`, `decision`, `production_change_mbpd`                                                                                                                                                               |

**Coverage:**

- `market_context`: May 2024 → May 2026 (hourly, DXY+VIX 99.9% coverage, EIA events aligned)
- `articles`: May 2024 → May 2026

### Notebook 05 rewrite — CSV → SQLite (14 May 2026)

**What changed:**

- All reads now come from DB (`articles`, `market_context`, `llm_features`) — no CSV reads or writes
- Replaced slow `O(n²)` `.apply(get_next_trading_hour)` loop with vectorized `merge_asof(direction='forward')`
- Inner join articles → market_context: articles beyond coverage are dropped and counted
- Left join `llm_features` → `has_llm_features` flag added; articles without LLM scores are kept, not dropped
- `liquidity` now stores LLM feature columns inline (denormalized for TFT training — one table to read)
- Sanity check: asserts aligned ≥ 13,690 before writing; aborts on merge errors or NULL `log_volume`

**Results after rewrite (full run, 22,795 articles):**

- Total articles in DB: 22,795
- Aligned to market_context: **22,795** (0 dropped — all fall within May 2024 → May 2026 window)
- Contemporaneous (<2h gap): **15,290**
- Forward-assigned (≥2h, off-hours/weekends): **7,505**
- With LLM features: **12,024**
- Without LLM features (pending extraction): **10,771**
- Date range: 2024-05-13 → 2026-05-12

### Next steps

- Run nb04 one more time to scrape remaining 2,749 pending articles
- Run nb06 (LLM feature extraction) for the ~10,771 articles without features
- Re-run nb05 after LLM extraction to refresh `liquidity` with new feature coverage
- Re-train TFT on Colab with expanded dataset (~22k aligned rows vs ~10k before)

---

## Phase 5 — LLM Extraction v2: Infrastructure and Calibration Setup (14 May 2026)

### Archival

- `02_notebooks/06_finbert_sentiment.ipynb` → `02_notebooks/old_stuff/06_finbert_sentiment.ipynb`
- `02_notebooks/06_llm_features.ipynb` → `02_notebooks/old_stuff/06_llm_features_v1.ipynb`
- Both preserved as reference for thesis methodology comparison (FinBERT vs LLM)

### Shared extraction module — `03_src/nlp/llm_features.py`

- `SYSTEM_PROMPT`: system-level instruction with inline event_type categories
- `EXTRACTION_TOOL`: Anthropic tool schema enforcing types and enums at the API level
- `extract_features(title, body, client) -> dict`: single-article extraction function
- **Schema changes from v1:**
  - `price_direction` removed (derivable from `sentiment_score` via thresholding)
  - `event_type` is now an array of 1–3 ordered categories (was single string); `"inventory"` dropped
  - `time_horizon` values: `immediate / short_term / structural` (was `long_term`)
  - `usable` (boolean) added as the only required field — short-circuits all other fields when false
  - Body truncated at 1500 chars (was 800)
- Tool-use API with `tool_choice` forcing `extract_article_features` — no JSON parsing
- Prompt caching on system prompt (`cache_control: ephemeral`)
- Model: `claude-haiku-4-5`, `max_tokens=400`

### Calibration infrastructure

- `02_notebooks/11_calibration.ipynb`: one-shot validation notebook over `calibration_sample` (30 articles)
  - Reads from DB, calls `extract_features` synchronously, saves to JSON — does NOT write to `llm_features`
  - Output: `05_reports/calibration/llm_calibration.json`
  - Estimated cost: ~$0.03
- `02_notebooks/06_llm_features.ipynb`: placeholder, pending calibration approval

### Next steps

- Run `11_calibration.ipynb` and review output JSON against hand-scored values
- Approve prompt → populate `06_llm_features.ipynb` for production batch (~20k articles)
- Run nb04 final pass (2,749 articles still pending body scrape)

### Calibration v2 — channel decomposition validated (18 May 2026)

- Re-ran 30-article calibration with extended schema (added `supply_impact`, `demand_impact`, `risk_premium` to existing fields).
- Output: `05_reports/calibration/llm_calibration_v2.json` (Haiku) and reference scores from GPT-4 family on same prompt.
- `usable` agreement: 27/30 (90%), up from 26/30 (87%) in v1. Three Haiku-rejected articles GPT accepted, all borderline corporate/analyst commentary. Acceptable.
- Sentiment correlation jumped from 0.39 (v1) to 0.88 (v2). Sign disagreements dropped from 4/13 to 1/14.
- Per-channel correlation (Haiku vs GPT):
  - `supply_impact`: r=0.94, MAD 0.11, 0 sign disagreements
  - `demand_impact`: r=0.96, MAD 0.05, 0 sign disagreements
  - `risk_premium`: r=0.82, MAD 0.13, 1 sign disagreement
- Within-model orthogonality preserved (all pairwise |r| < 0.5 in both models).
- Haiku-specific weakness: 3/14 usable articles scored all three channels = 0.0 (vs 0/14 in GPT). Borderline articles where Haiku punts. Not blocking — flag in thesis writeup.
- Decision: prompt validated. Proceed to full batch on all articles with usable bodies (~19,600).
- Cost projection for full batch: $15–20 with Batches API + system prompt caching.

### Pending: full LLM extraction batch

- Target: all articles where `body IS NOT NULL AND body NOT LIKE 'ERROR%'` (~19,600 rows after scraping completion).
- Filter strategy: send everything with a real body, let LLM `usable` flag do the final filtering. `body_valid` heuristic deprecated as canonical filter.
- Use Anthropic Batches API + prompt caching. Add retry/error handling to batch runner before kickoff.
- Write to `llm_features` (already migrated schema: 8 nullable columns + `usable BOOLEAN`, `article_id` PK).
- Post-extraction: entity normalization pass (canonical-name dict mapping variants → standard forms).

---

## Phase 5 — Full LLM extraction batch (2026-05-18)

### Batch details

- Batch ID: `msgbatch_018JvAUYvymcoVVakpqA3f1T`
- Model: `claude-haiku-4-5`, Batches API (50% discount), prompt caching on system prompt
- SQL filter: `body IS NOT NULL AND body NOT LIKE 'ERROR%'` — no body_valid filter
- Articles submitted: 107
- Successful results: 107
- Errors: 0 (logged to `05_reports/calibration/batch_errors.json`)

### Results written to llm_features

- Total rows: 19,619
- usable=true: 11,675
- usable=false: 7,944

### Channel stats (usable articles)

- sentiment_score: mean=0.025, std=0.501
- supply_impact: mean=-0.071, std=0.463
- demand_impact: mean=-0.075, std=0.34
- risk_premium: mean=0.156, std=0.426
- All channels = 0.0: 1,161 articles

### Actual cost

- Final batch (107 articles): $0.1501
- Two earlier batches (interrupted by credit limits): credit-funded
- **Aggregate cost across three batch submissions: $38.82**
- Higher than pre-batch estimate (~$7.81 scaled from 107-article cell) due to:
  - Prompt-cache strategy did not fire across separate batch submissions (each batch re-paid system-prompt write cost)
  - Body truncation raised to 1,500 chars (from 800 in initial estimate), nearly doubling input tokens per article
  - Per-usable-article output tokens higher than the 200-token assumption because channel decomposition adds output fields
- Cost remained operationally negligible. Cost mitigation strategies (Batches API 50% discount, short-circuit on usable=false, prompt caching design) remain documented in `thesis_decisions_log.md`.

### Next steps

- Inspect batch_errors.json; resubmit failed articles if meaningful in volume
- Run nb05 (alignment) to refresh liquidity table with new usable-filtered feature coverage
- Re-train TFT on Colab with expanded dataset

---

## Phase 5 — Notebook 05 DB Migration (2026-05-18)

### Alignment rewrite — CSV → SQLite

**Changes:**

- All reads now come from `wti_thesis.db` (articles, market_context, llm_features) — no CSV reads
- Replaced slow `apply(get_next_trading_hour)` loop with vectorized `merge_asof(direction='forward')`
- Inner join articles → market_context: articles beyond coverage are dropped and counted
- Left join llm_features → `has_llm_features` flag added
- Sanity check: assert aligned >= 13,690 before writing
- `liquidity` table now includes LLM feature columns (denormalized for TFT training)
- `DROP TABLE IF EXISTS liquidity` before write — no stale rows

**Results after rewrite:**

- Total articles in DB: 22,795
- Aligned to market_context: 22,795
- Dropped (out of range): 0
- Contemporaneous (<2h gap): 15,290
- Forward-assigned (>=2h): 7,505
- With LLM features: 19,619
- Without LLM features: 3,176

**TODO before TFT v2 training: notebook 05 currently does not propagate DXY and VIX from market_context to liquidity. Add these columns to the alignment step before training, or the TFT v2 macro covariates will be empty.**

**TODO PIPELINE to create the entities list for the dataset, rightnow there are different ways of naming a single entity**

**TODO refine llm judged articles that may include false positives like EV industry**

## TFT v2 Training — Prep Plan and TODO

### Status snapshot

- **Thesis writing**: Chapter 3 complete and reviewed by Lejla. Chapter 4 §4.2 (Phase 1) and §4.3.1–§4.3.6 (Phase 2 except TFT v2) drafted and verified. Outline updated to reflect the FinBERT/Haiku phase boundary.
- **Blocked on TFT v2 training**: §4.3.7, §4.3.8, §4.4 (partially), Chapter 5, Chapter 6.

### Locked decisions

**Prediction horizon**: multi-horizon `[1, 3, 6, 12, 28]` hours. Single model, one forward pass produces all horizons. The 28h anchor tests the v1 daily-memory attention finding at lag −27/−28h. Honest framing required in §4.3.7: absolute accuracy at 28h will be poor; what matters is the _shape_ of the per-horizon error curve.

**Target variables**: multi-target `[log_volume, amihud, price_range]`. TFT supports this natively. Training cost ~5x v1.

**Train/val/test split**: 70/15/15 temporal. Real held-out test set, touched once at the end. With ~11,000+ hours: ~7,700 train / ~1,650 val / ~1,650 test.

**Compute**: Colab Pro upgrade (A100/L4/V100 instead of T4). ~5–10× speedup. ~$10/month, cancel after.

**Ablation strategy**: iterative with sub-version names (v2.0, v2.1, v2.2...). All variants use Haiku v2 features (channels + composite sentiment + magnitude + certainty + entities + event_type + time_horizon).

Default 3-training plan:

1. **v2.0 baseline** — Haiku v2 features used in a v1-style minimal way: int-encoded categoricals (event_type, time_horizon), no entity flags, single horizon (1h), single target (log_volume). Tests: "does the channel decomposition alone help, when used as v1 used composite sentiment?"
2. **v2.1 proper categoricals** — v2.0 + proper categorical encoding for `event_type` and `time_horizon` (not int-encoded). Single horizon, single target. Tests: "do proper categoricals add signal on top of channels?"
3. **v2.2 full** — v2.1 + multi-hot entity flags (top 15–20 canonical) + multi-horizon `[1, 3, 6, 12, 28]` + multi-target `[log_volume, amihud, price_range]`. Production v2, subject of §4.3.7.

Can expand to 5 ablations if 3 don't cleanly attribute improvements. Infrastructure should support both.

**Success criteria (set in advance)**:

1. Channels are economically interpretable in attention/importance (qualitative)
2. Directional asymmetry test reaches p<0.05 (quantitative RQ2)
3. Multi-horizon error structure matches lag OLS peak at +6h (RQ1)

Scoring: 3/3 strong; 2/3 solid; 1/3 partial; 0/3 negative result, all discussed honestly in Chapter 5.

### Engineering prep tasks (do in order, before any training)

**Task 1 ✅ — Fix notebook 05 to propagate DXY/VIX into `liquidity`**. (completed 2026-05-30)
The DB migration accidentally dropped these. In notebook 05, add `dxy` and `vix` to both the `market_context` column selection and the final `df_liquidity` projection. Re-run notebook end-to-end. Verify with:

```sql
SELECT COUNT(*),
       SUM(CASE WHEN dxy IS NOT NULL THEN 1 ELSE 0 END) AS with_dxy,
       SUM(CASE WHEN vix IS NOT NULL THEN 1 ELSE 0 END) AS with_vix
FROM liquidity;
```

Both should be near 22,795.

**Task 2 ✅ — Entity normalization** (`02_notebooks/12_entity_normalization.ipynb`). (completed 2026-06-01)
Materialize `raw_entity_counts` from `llm_features.entities`, triage by frequency, maintain `ENTITY_CANONICAL` / `CANONICAL_ENTITIES` in `llm_features.py`, write `article_entities` (long-format article_id × canonical_entity). Entity flags (top canonicals) feed TFT v2.2.

**Task 3 ✅ — EV/off-topic cleanup heuristic**. (completed 2026-06-01)
Manual audit (§4.3.3) found LLM `usable=true` includes some off-topic articles. Find them:

```sql
SELECT a.id, a.title, lf.sentiment_score, lf.event_type
FROM articles a
JOIN llm_features lf ON a.id = lf.article_id
WHERE lf.usable = 1
  AND lf.supply_impact = 0
  AND lf.demand_impact = 0
  AND lf.risk_premium = 0
  AND lf.sentiment_score != 0
ORDER BY ABS(lf.sentiment_score) DESC
LIMIT 50;
```

Recommendation: flag with `usable_strict` column, don't drop. Allows A/B in ablation.

**Task 4 — Lock train/val/test indices**.
Once `liquidity` is refreshed, compute split points based on final hourly grid size. Save as constants in the training notebook so all ablation variants use identical splits.

**Task 5 — Scaffold TFT v2 training notebook** `02_notebooks/13_tft_v2_training.ipynb`.
Use a single `ABLATION_VARIANT` constant (`'v2.0'`, `'v2.1'`, `'v2.2'`) at the top that gates which features are included. Avoids three near-identical notebooks. Structure:

- Load liquidity + llm_features (with new entity flags)
- Apply EV cleanup if enabled
- Construct `TimeSeriesDataSet` per variant
- Train, checkpoint to `01_data/models/tft_v2_X.ckpt`
- Mirror notebook 10 analysis: per-target/horizon loss, VSN importance, attention by sentiment direction, RQ2 t-test

### Open questions for training session

- Final thesis-facing naming (v2.0/v2.1/v2.2 working; may collapse to "TFT v2" with intermediate ablations as prose mentions in §4.3.7)
- 3 vs 5 ablation runs — decide based on what data shows
- §4.3.7 vs §4.3.8 split — likely §4.3.7 = ablation trajectory, §4.3.8 = v1 vs v2 headline comparison. Decide after results.
- §4.4 alignment robustness check (ceiling vs floor on Phase 1 lag OLS): independent of v2, ~30 min experiment. Could be a warm-up.

### What's NOT in this TODO

Chapter 2 Background drafting; Chapter 1 Introduction; Appendix assembly; Discussion/Conclusion drafting; supervisor feedback integration. Separate workstream notes if needed.

---

## Phase 5 — Notebook 05 alignment run (2026-05-30)

### Alignment results

**Schema:**

- `liquidity` includes: `usable`, `supply_impact`, `demand_impact`, `risk_premium`, `dxy`, `vix`
- `price_direction` dropped; `body_valid` no longer propagated (superseded by `usable`)
- `usable=1` is the canonical downstream filter for modeling

**Pipeline:**

- All reads from `wti_thesis.db` — no CSV reads
- Vectorized `merge_asof(direction='forward')` replaces the old `.apply()` loop
- Left join `llm_features` → `has_llm_features` and `usable` flags
- Sanity check: assert aligned >= 22,000 before writing
- `DROP TABLE IF EXISTS liquidity` before write — no stale rows

**Results:**

- Total articles in DB: 22,795
- Aligned to market_context: 22,795
- Dropped (out of range): 0
- Contemporaneous (<2h gap): 15,290
- Forward-assigned (>=2h): 7,505
- Modeling-ready (usable=1): 11,675 ← canonical filter
- LLM-rejected (usable=0, LLM called): 7,944
- No-body (no LLM called): 3,176

**Channel coverage:**

- supply_impact, demand_impact, risk_premium non-null iff usable=1 (11,675 rows)
- All three channel counts verified equal to usable=1 count

---

## TFT v2 Prep — Task 1: DXY/VIX propagated to liquidity (2026-05-30)

**Change:** `dxy` and `vix` were missing from `liquidity` because the Phase 5 DB migration
selected only `log_volume, price_range, log_return, amihud` from `market_context`.
Fixed by adding both columns to:

1. `market_context` SELECT in the load cell
2. `merge_asof` column projection in the assign cell
3. `df_liquidity` column list in the write cell

**Verification (2026-05-30):**

```sql
SELECT COUNT(*),
       SUM(CASE WHEN dxy IS NOT NULL THEN 1 ELSE 0 END) AS with_dxy,
       SUM(CASE WHEN vix IS NOT NULL THEN 1 ELSE 0 END) AS with_vix
FROM liquidity;
```

- total: 22,795 | with_dxy: 22,794 | with_vix: 22,788 ✅
- Gap of 1 / 7 rows matches the ~0.1% off-hours coverage holes in market_context.

**Next:** Task 3 — EV/off-topic cleanup heuristic (`usable_strict` column).

---

## TFT v2 Prep — Task 2: Entity normalization (2026-06-01)

### Approach

Materialized the full raw entity distribution from `llm_features.entities` into `raw_entity_counts` (5,807 distinct strings, 44,774 total mentions). Triaged in three passes with a 25-occurrence minimum threshold for new canonicals:

- **≥100 occurrences (61 strings):** 53 already mapped; 7 noise/geo/news-agency strings dropped (Reuters, Europe†, AAA Oregon/Idaho, Washington, Asia†, Oregon, NYMEX). Goldman Sachs kept in variant map but excluded from canonical list (analyst-commentary signal muddled across bull/bear coverage).
- **50–99 occurrences (34 strings):** 15 already mapped; 4 variant fixes (Opec+, President Donald Trump, West Texas Intermediate (WTI), Tehran → Iran); 6 new canonicals (UK, BP, Algeria, Germany, Australia, Persian Gulf); Philippines and Malaysia dropped (demand signal already captured by Asia/China/Japan).
- **25–49 occurrences (71 strings):** 2 bug fixes (EU and API strings not mapped despite being canonicals); 12 variant fixes (Houthi, Aramco, Beijing → China, Moscow → Russia, Brent Crude, Opec, Abu Dhabi → UAE, Fatih Birol → IEA, IRGC → Iran, Eurozone → EU, Xi Jinping → China, Narendra Modi → India); 9 new canonicals (ExxonMobil, Rosneft, Lukoil, TotalEnergies, Türkiye, Permian Basin, Hungary, Egypt, Gulf of Mexico); ConocoPhillips and Indonesia dropped.
- **Case/punctuation sweep:** Added Opec, Opec+, OPEC Plus, Brent Crude, United States of America, Russian Federation.

† Europe and Asia reinstated as canonicals (geographic aggregates parallel to Middle East).

**Biden:** Combined count 92 (above 80 threshold) → added as separate canonical. Asymmetric treatment relative to Xi Jinping and Modi (both mapped to their countries) is defensible: Trump's exceptional frequency (1,935) creates the precedent for a standalone US-president canonical, and Biden's count is high enough to warrant the same treatment for post-administration coverage continuity.

**DOE:** "Department of Energy" contaminated by Philippines DoE; clean prefixed variants below threshold → dropped. Rationale in `thesis_decisions_log.md`.

### Final state

- `03_src/nlp/llm_features.py`: **208 variant mappings** → **71 canonical entities** (up from 52)
- `wti_thesis.db`: `raw_entity_counts` materialized (5,807 rows); `article_entities` written (29,473 rows, 10,293 distinct articles, 71 distinct entities used)
- Coverage: 169 / 5,807 distinct raw strings mapped (2.9%); 30,135 / 44,774 mention-level coverage (67.3%)
- Notebook: `02_notebooks/12_entity_normalization.ipynb`

**Next:** Task 3 — EV/off-topic cleanup heuristic (`usable_strict` column).

## TFT v2 Prep — Task 3 supporting analysis: zero-channel breakdown (2026-06-01)

### Diagnostic counts (all from `llm_features WHERE usable=1`)

Total zero-channel articles (all 3 channels = 0, sentiment_score != 0): **1,161**

Breakdown by event_type pattern:

| Subset                                                                                       | Count | Treatment                  |
| -------------------------------------------------------------------------------------------- | ----: | -------------------------- |
| Inconsistent: channel-relevant event_type (supply/demand/geopolitical) but channels all zero |   242 | `usable_strict = 0`        |
| By-design: technical-only event_type, zero channels appropriate                              |   243 | `usable_strict = 1` (kept) |
| Macro/other/structural: intermediate event_types, no specific channel implication            |   676 | `usable_strict = 1` (kept) |

Rationale: only the 242 "inconsistent" cases represent residual LLM punting (the calibration weakness from §4.3.6). The 919 zero-channel articles in the other two categories are substantive on-topic content where the channels are correctly zero by article type (pure price reports, technical analysis, equity-market reactions to oil moves, macro framing without specific supply/demand implication).

Original EV/off-topic contamination hypothesis was not confirmed by this dump — the manual audit found no EV stories or scraping artifacts in the top 50 zero-channel articles.

### Final state

- 11,675 articles with `usable=1`:
  - 11,433 with `usable_strict=1`
  - 242 with `usable_strict=0`
- 7,944 articles with `usable=0` (unchanged; `usable_strict` also 0)

**Next:** Task 4 — Lock train/val/test indices.

## TFT v2 Prep — Task 4: Lock train/val/test indices (2026-06-02)

### Approach

Temporal 70/15/15 split with 48h encoder buffer between splits to prevent encoder-window leakage across boundaries.

### Verified date ranges

- Train: 2024-05-13 11:00 UTC → 2025-10-07 09:00 UTC (7,862 hours, ~17 months)
- Val: 2025-10-09 12:00 UTC → 2026-01-27 18:00 UTC (1,637 hours, ~3.7 months)
- Test: 2026-01-29 21:00 UTC → 2026-05-13 11:00 UTC (1,637 hours, ~3.5 months)

### Regime context

- June 21, 2025: US strike on Iranian nuclear facilities (in TRAIN set)
- Feb 28, 2026: US/Israel attack on Iran, war onset (in TEST set, first post-onset hour at row idx 10056 = 2026-03-01 23:00 UTC)

Test set spans the regime change: first ~430 hours pre-war, remaining ~1,200 hours during the war. Test metrics to be reported on both slices separately to evaluate cross-regime generalization. This is an emergent property of the temporal split, not a designed contrast.

### Final state

- Constants saved to `03_src/tft/config.py`
- All Task 5 ablation variants will import from this single source

**Next:** Task 5 — Scaffold TFT v2 training notebook `02_notebooks/13_tft_v2_training.ipynb`.

## TFTv2 Training process

### TFT model parameter comparison

| Model    |    Parameters | Notes                                                                                                                                                                         |
| -------- | ------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TFT v1   |       112,685 | FinBERT-mapped sentiment + magnitude + int-encoded categoricals + DXY/VIX. Architecture: hidden_size=32, attention_head_size=4, dropout=0.1, hidden_continuous_size=16.       |
| TFT v2.0 |       118,114 | Same architecture as v1. Adds three channel features (supply_impact, demand_impact, risk_premium). +5,429 params from extended VSN feature list.                              |
| TFT v2.1 | TBD after run | Same as v2.0 but proper categorical encoding for event_type and time_horizon (replaces int-encoded reals). Small embedding tables add ~few hundred params.                    |
| TFT v2.2 | TBD after run | Same as v2.1 plus 71 entity flag columns, multi-target output heads (3 targets), multi-horizon decoder (max prediction length 28). Substantially larger; expected ~200k-300k. |

Parameter counts confirm architectural constancy across the v1 → v2.0 → v2.1 transition: identical hidden_size, attention_head_size, dropout, and hidden_continuous_size. The v1-vs-v2.0 difference (~5%) is attributable entirely to the three new channel features entering the Variable Selection Network. The v2.2 expansion reflects the entity-flag features and the multi-target/multi-horizon output configuration.

## TFT v2.0 — first ablation run results (2026-06-02)

### Training

- Best checkpoint: `tft_v2.0-epoch=19-val_loss=0.2941.ckpt`
- Best val_loss: 0.2941 at epoch 19
- Early-stopped at epoch 25 (max_epochs=50, patience=5)
- Compute: Colab T4 GPU
- Trainable parameters: 118,114
- Architecture: identical to TFT v1 (hidden_size=32, attention_head_size=4, dropout=0.1, hidden_continuous_size=16)

### Test set metrics (held-out, 1,637 hours, Jan 29 – May 13, 2026)

| Slice                            |     N |   MAE |  RMSE |
| -------------------------------- | ----: | ----: | ----: |
| Val                              | 1,637 | 0.466 | 1.035 |
| Test full                        | 1,637 | 0.397 | 0.696 |
| Test pre-war (before 2026-02-28) |   461 | 0.372 | 0.621 |
| Test war (from 2026-02-28)       | 1,176 | 0.406 | 0.723 |

### Persistence baseline (predict log_volume(t+1) = log_volume(t))

| Slice        | Persistence MAE | TFT v2.0 MAE | Reduction |
| ------------ | --------------: | -----------: | --------: |
| Val          |           1.171 |        0.466 |       60% |
| Test full    |           1.071 |        0.397 |       63% |
| Test pre-war |           1.072 |        0.372 |       65% |
| Test war     |           1.071 |        0.406 |       62% |

### Observations

- TFT v2.0 substantially beats persistence on every slice. 60-65% MAE reduction confirms the model is learning real signal from input features, not just memorizing autocorrelation.
- War-slice MAE is ~9% worse than pre-war-slice MAE (0.406 vs 0.372), but only ~5% worse than full test MAE. The model generalizes reasonably across the regime change.
- Val MAE (0.466) is worse than test MAE (0.397) despite identical sample size. Attributed to val period (Oct 2025 – Jan 2026) having higher persistence baseline MAE (1.171 vs test's 1.071), i.e., genuinely higher volume variability that's harder to predict.
- Direct comparison of v2.0's val_loss (0.294) to TFT v1's val_loss (0.204) is not apples-to-apples because v1's val set covers a different period (Phase 1 corpus, 80/20 split). The persistence-baseline comparison is the more meaningful sanity check on v2.0's behavior.

### Pending for v2.0 (Cells 14-17)

- Feature importance ranking (does sentiment_score still dominate or do channels carry weight?)
- Attention weight peak lag (does v1's -4h peak reproduce?)
- Directional asymmetry: bearish vs bullish predicted volume t-test
- Save artifacts to `04_outputs/tft_v2/v2.0/`

### Next

- Complete v2.0 evaluation (Cells 14-17)
- Run v2.1 ablation (proper categoricals)
- Run v2.2 ablation (entities + multi-horizon + multi-target)
