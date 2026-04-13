# Project Logbook — Modeling News-Driven Liquidity Dynamics in Commodities Markets

**Student:** Enrique Favila Martínez | **Institution:** Radboud University MSc AI | **Host:** Hammer Market Intelligence | **Supervisor:** Dr. Lejla Batina | **Last updated:** April 2026

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
- `log_return` — log(Close_t/Close_{t-1})
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

| Lag | Bearish | Bullish |
|---|---|---|
| +1h | ✅ sig | ✅ sig |
| +2h | ❌ | ❌ |
| +3h | ❌ | ✅ sig |
| +4h | ✅ sig | ✅ sig |
| +6h | ✅ sig (peak) | ✅ sig (peak) |
| +8h | ❌ | ❌ |
| +12h | ❌ | ✅ sig negative |

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