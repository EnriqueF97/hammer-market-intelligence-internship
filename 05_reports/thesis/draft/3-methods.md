# Chapter 3 - Methods

## 3.1 Data sources and acquisition

This thesis combines three distinct data sources into a unified hourly dataset for WTI crude oil: market data on prices and trading activity, structured macroeconomic indicators, and unstructured news content. Each source has its own native cadence and access pattern, and harmonising them onto a common time grid is a central methodological concern, addressed in Section 3.2.

### 3.1.1 Commodity selection

WTI crude oil futures were chosen as the target commodity after preliminary screening of candidates. Sugar futures were considered initially but excluded due to limited price variability and weak responsiveness to news events at hourly resolution, both of which are essential for the research questions of this thesis. WTI, by contrast, exhibits rich event structure driven by geopolitics, OPEC policy, the weekly EIA inventory cycle, and broader macroeconomic forces, all of which produce measurable hourly volume and price reactions.

### 3.1.2 Market data

Hourly OHLCV data are retrieved via the yfinance Python library [N] for three instruments:

CL=F: WTI crude oil front-month futures contract, the primary target series
DX-Y.NYB: the U.S. Dollar Index (DXY), used as a macroeconomic covariate
^VIX: the CBOE Volatility Index, used as a market-stress covariate

DXY and VIX enter as exogenous controls in the Phase 2 modeling pipeline (Section 4.3), allowing the news signal to be isolated from broader macro-financial dynamics. They are not used in the Phase 1 statistical analyses (Section 4.2), which focus on the direct relationship between news sentiment and trading volume without macro controls.
From the raw OHLCV records, we derive three liquidity proxies and one return measure at hourly resolution:

log*volume: the natural logarithm of contracts traded per hour, used as the primary liquidity measure
price_range: the hourly Parkinson range, computed as log(high) − log(low), a high-frequency volatility proxy
log_return: the hourly log return, log(close_t / close*{t-1})
amihud: the Amihud illiquidity ratio at hourly resolution, computed as |log_return| / volume following [Amihud, N]. The ratio is undefined for hours with zero volume; such rows are excluded from analyses where Amihud is the dependent variable.

Coverage spans 13 May 2024 to 13 May 2026, yielding 11,232 hourly observations across approximately 24 months of trading. In Phase 1 (Section 4.2.1), the analysis used a narrower window of approximately March 2024 to February 2026 covering the FinBERT-extracted news corpus. Coverage was extended in Phase 2 (Section 4.3.1) to incorporate news from the post-war period through May 2026.

### 3.1.3 EIA inventory data

Weekly U.S. commercial crude oil inventory reports are retrieved from the U.S. Energy Information Administration (EIA) v2 API [N]. We query the WCRSTUS1 series (weekly ending stocks of crude oil and petroleum products, U.S. total), specifying `frequency=weekly`. The full historical series from January 2020 onward is downloaded to allow computation of rolling baselines, even though only the 2024–2026 window is used in modeling.

Each weekly release is converted into two derived features available at hourly resolution after temporal alignment:

- `is_eia_release`: a binary indicator that fires on the trading hour containing or immediately following the official release time
- `eia_surprise`: the weekly inventory change relative to a rolling four-week baseline, capturing the directional surprise component of each release

EIA releases occur on Wednesdays at 10:30 a.m. U.S. Eastern Time and are among the most consistently market-moving scheduled events in WTI trading [N]. Including the surprise component as a continuous feature, rather than only the binary release flag, allows the models to differentiate large draws from small or expected ones.

### 3.1.4 News data

Unstructured news content is sourced from the GDELT Project [N], a large-scale machine-monitored news corpus covering broadcast, print, and web sources worldwide. We use the GDELT 2.0 Document API to retrieve article metadata (URL, source, publication time, title) across a set of thematic queries chosen to capture WTI-relevant content.

The query set evolved across the two methodological phases. In Phase 1, the focus was on direct oil-market terms, yielding eight queries:

```
crude oil market
OPEC production cut
oil supply demand
petroleum inventory
Brent crude price
oil inventory report
energy market outlook
crude oil WTI price
```

The Phase 1 scrape produced 51,948 raw article records across the March 2024 to February 2026 window. After deduplication and English-only filtering, 16,326 unique articles remained. Article body content was retrieved by HTTP-fetching each URL and parsing the page with BeautifulSoup [N]. Body retrieval achieved an approximate 80% response rate, with failures concentrated on paywalled domains, Cloudflare-protected sources, and pages requiring JavaScript rendering. An allow-list heuristic (Section 3.4) classified 7,756 of the retrieved bodies as substantive content. The Phase 1 analyses operated on a combined dataset of 13,690 articles: 7,756 with valid bodies feeding title-plus-body sentiment extraction, and 5,934 articles where body retrieval failed but the title alone was retained as a fallback input.

In Phase 2, the query set was expanded to capture geopolitical and demand-side themes that became prominent in the post-war coverage period (Section 4.3.1):

```
Iran sanctions oil
Saudi Arabia oil production
China oil demand
Russia oil exports
oil supply disruption
```

The expanded scrape was extended through May 2026, raising the total corpus to 22,795 article records spanning January 2024 to May 2026. Across the full Phase 2 corpus, 19,619 articles produced a real body; the remaining 2,749 yielded either a null response, a Cloudflare or paywall block, or an explicit fetch error. The choice to forward articles without bodies into LLM-based filtering rather than retain a title-only fallback path is discussed in Section 3.4.

### 3.1.5 Persistence

All retrieved data are stored in a single SQLite database (`wti_thesis.db`) with the following primary tables: `articles` (raw GDELT records and scraped bodies), `market_context` (hourly OHLCV and derived liquidity features for WTI, DXY, and VIX), `eia_events` (raw EIA releases), `llm_features` (LLM-extracted news features, Phase 2 only), and `liquidity` (the modeling-ready joined view, populated by the alignment step in Section 3.2). Storing all data in a single relational database, rather than in CSV intermediates as in the original Phase 1 pipeline, eliminated state drift across notebooks and made the canonical alignment computable as a single deterministic join.

## 3.2 Temporal alignment

The three data sources operate on different clocks. Hourly market data lives on a clean grid of trading hours, with each hourly bar tagged by its closing timestamp. EIA releases are weekly events at fixed times. News articles, however, can be published at arbitrary timestamps including overnight hours, weekends, and holidays when no trading is taking place. To support models that condition liquidity at hour `t` on news published before `t`, each article must be assigned to a specific hour in the market grid.

We address this with two design choices, applied at insertion time when articles are written to the database: ceiling assignment to the next trading hour, and forward-assignment of off-hours articles to the next available trading hour.

### 3.2.1 Ceiling assignment to the next trading hour

For each article with publication timestamp `τ`, we compute its assigned market hour as

```
datetime_hour(τ) = ceil(τ, '1h')
```

That is, an article published at 14:23 UTC is assigned to the 15:00 hourly bar, not the 14:00 bar. The motivation is causal: the 14:00 hourly bar covers the period from 14:00 to 15:00, and its volume is partially formed before 14:23. Assigning the article to the 14:00 bar would create a reverse-causal contamination, allowing news to influence a candle whose formation preceded the news publication. The ceiling rule guarantees that the article strictly precedes the start of the trading hour it is matched to.

An earlier version of the pipeline used `round` rather than `ceil`, which produced the same causal leak for articles published in the second half of an hour. The fix was applied early in the project and propagated through all downstream notebooks.

The ceiling rule is conservative: it assigns news to the next full hour even when the article would plausibly have moved markets within minutes of publication. This may underestimate the immediate (intra-hour) reaction. However, it eliminates the alternative bias of attributing pre-publication price movements to post-publication news, which is the more serious threat to causal inference. As a robustness check, Section 4.4.2 discusses re-running the lag analysis with floor alignment and reports whether the lag peak shifts as a result.

### 3.2.2 Forward-assignment of off-hours articles

A substantial portion of the news corpus is published during hours when no trading is taking place: overnight, on weekends, or during holiday closures. Two strategies were considered. The first is to drop such articles entirely. The second is to forward-assign them to the next available trading hour. We adopted the second.

The motivation is empirical. Geopolitical events, supply shocks, and weekend OPEC communiqués are routinely covered during off-hours and are precisely the events that drive the largest market reactions when trading resumes. Dropping them would systematically remove the most informative subset of the corpus. Forward-assignment, by contrast, preserves the information at the cost of introducing a measurement-time lag between publication and the assigned hour.

Concretely, for each article whose ceiling-assigned hour does not appear in the market grid (because the market is closed), we assign the article to the next available `market_context` row in chronological order. This is implemented as a vectorised pandas `merge_asof` join with `direction='forward'`, which performs the lookup efficiently across the full corpus in a single pass.

We retain a diagnostic column, `assignment_gap`, defined as the difference in hours between the article's assigned market hour and its original publication timestamp:

```
assignment_gap = datetime_hour − publication_timestamp
```

For articles published during a trading session, this is at most one hour. For weekend or overnight articles, it can extend to many hours or, in the worst case, across a multi-day holiday closure. The gap is preserved in the `liquidity` table and is used downstream both as a control variable and as a basis for stratified analyses that distinguish _contemporaneous_ news (gap < 2h) from _forward-assigned_ news (gap ≥ 2h). The Phase 1 VAR analysis (Section 4.2.5) uses only contemporaneous articles, aggregated to hourly resolution; the TFT models in Phase 2 use the full corpus and let the model learn to weight contemporaneous and forward-assigned articles differently if the data support such a distinction.

### 3.2.3 Joined output

The alignment step produces the `liquidity` table, in which each row corresponds to one article enriched with the market-context features at its assigned hour, plus the EIA release flags and surprise value at the same hour. Articles whose ceiling-assigned hours fall outside the market coverage window (May 2024 to May 2026) are dropped via an inner join, eliminating a small number of pre-coverage articles that cannot be matched to market data.

The resulting `liquidity` table has 22,795 rows, of which 15,290 are contemporaneous (gap < 2h) and 7,505 are forward-assigned (gap ≥ 2h). These two subsets are tracked separately in some analyses but combined in the primary modeling.

## 3.3 News feature extraction approaches

News articles are unstructured text; downstream statistical and deep learning models require numerical features. The transformation from raw article body to model-ready features is the central NLP problem of this thesis. We use two distinct approaches across the two methodological phases of the project: FinBERT-based sentiment classification in Phase 1, and structured LLM extraction with Claude Haiku in Phase 2. The two approaches differ both in their output representation and in their conceptual model of what a "news feature" is.

### 3.3.1 FinBERT sentiment extraction

In Phase 1, we use FinBERT [N], a BERT variant fine-tuned on financial text for three-class sentiment classification (positive, neutral, negative). FinBERT is among the most widely used sentiment models in financial NLP and provides a natural baseline for comparison.

The model is applied via the HuggingFace Transformers library [N], running on Apple Silicon GPU acceleration via the MPS backend. For each article, the model returns softmax probabilities over the three classes; the predicted label is taken as the argmax of these probabilities, and a scalar `confidence` value is recorded as the maximum class probability:

```
confidence = max(P_positive, P_negative, P_neutral)
```

The categorical label, all three class probabilities, and the confidence value are stored in the database for downstream use.

FinBERT is applied twice per article:

- **Title-only input**: the article title fed alone to the model
- **Title-plus-body input**: the title concatenated with the body, subject to FinBERT's 512-token input limit

Body content is truncated where necessary, with the title preserved at the start of the sequence so that the most salient signal is never lost to truncation. Articles for which body retrieval failed (Section 3.1) fall back to title-only sentiment as the canonical input, with the body-availability status preserved in the database so downstream analyses can distinguish the two input regimes.

For the lag OLS and VAR analyses (Sections 3.5 and 4.2), the title-plus-body categorical label is mapped to a numeric scale via `{positive: +1, neutral: 0, negative: -1}` and used directly as a discrete-valued `sentiment_score`. This simple mapping preserves the directional information FinBERT provides while making no use of the underlying probability magnitudes or the confidence value. The retention of the raw probabilities in the database enables alternative scoring schemes (such as a continuous score derived from `P(positive) − P(negative)`) but these were not explored in any Phase 1 analysis presented in this thesis. The title-only categorical label is reserved for the comparative analysis in Section 4.2.2 (the headline bias finding).

### 3.3.2 Structured LLM extraction with Claude Haiku

In Phase 2, FinBERT is replaced by Claude Haiku 4.5 [N], a smaller-scale member of Anthropic's Claude model family, used to extract a richer set of structured features from each article. The motivation for moving from FinBERT to an LLM is taken up in detail in Chapter 4 (Section 4.3.2); briefly, the three-class categorical output of FinBERT collapses information that is naturally continuous and multidimensional, and the confidence value (the maximum class probability) carries limited downstream signal.

The Haiku extraction underwent one schema iteration during Phase 2. The initial schema (referred to as Haiku v1) was used to produce features for the first TFT training (Section 4.3.5) on the Phase 1 corpus of articles. It returned, for each article: a continuous sentiment_score in [-1, +1], a magnitude score for event importance, a price_direction label, an event_type label, a list of entities, a certainty score, and a time_horizon label. The revised schema (referred to as Haiku v2) was introduced in response to an inter-model calibration finding documented in Section 4.3.4. It dropped the price_direction field, changed event_type from a single label to an array of one to three labels ordered by salience, introduced an explicit usable boolean for filtering (Section 3.4), and added three orthogonal economic channel scores (supply_impact, demand_impact, risk_premium) described in Section 3.3.4. The full Haiku v2 prompt and tool schema are reproduced in Appendix A.

### 3.3.3 Tool-use API and schema enforcement

The Haiku extraction uses Anthropic's tool-use API [N] with `tool_choice` forcing the model to invoke a single `extract_article_features` tool. This is not a stylistic choice; it is a methodological one. Tool-use enforces the output schema at the API level, including enumerated values for categorical fields, type constraints on numeric fields, and required-field validation. An earlier extraction pass that relied on the LLM to produce valid JSON via prompt instruction alone produced occasional schema violations (e.g., the model returning `slightly_bullish` instead of a numeric score, or `medium_term` instead of one of the three valid `time_horizon` values). With tool-use enforcement these failures become impossible at the API boundary, eliminating an entire class of parsing brittleness from the pipeline.

The tool-use design also enables a short-circuit optimisation. When the model determines that an article is unusable (a paywall block, a cookie notice, or off-topic content), only the `usable` field is required to be set; all other fields are optional. The model exploits this to return roughly 10 output tokens instead of 200 on unusable articles, reducing the cost of unusable-article extraction by roughly 90% at the per-article output level.

### 3.3.4 Channel decomposition

The Haiku schema includes three continuous scores beyond the composite sentiment value: `supply_impact`, `demand_impact`, and `risk_premium`, each in [-1, +1]. These three channels are designed to decompose the economic content of a news article into orthogonal dimensions:

- `supply_impact` measures the article's implication for the quantity of oil available to the market. Positive values indicate supply increases (sanctions lifted, OPEC adding barrels, new pipeline online); negative values indicate supply decreases (refinery outages, OPEC cuts, infrastructure attacks).
- `demand_impact` measures the article's implication for oil consumption. Positive values indicate demand strength (Fed easing, China stimulus, growth surprise); negative values indicate demand weakness (recession signals, Fed hawkishness, lockdowns).
- `risk_premium` measures the change in geopolitical or operational risk priced into oil. Positive values indicate escalation (military strikes on oil infrastructure, sanctions imposed); negative values indicate de-escalation (ceasefire, sanctions lifted).

The three channels were not part of the original Phase 2 schema. They were introduced as a response to a documented inter-model calibration failure on the composite `sentiment_score`, in which Haiku and a GPT-family reference model disagreed on the sign of sentiment for high-magnitude geopolitical events. The full reasoning, calibration evidence, and design choices behind the decomposition are documented in Chapter 4 (Section 4.3.5). The methodological role of the channels in this thesis is twofold: they are higher-resolution features for the TFT v2 model (Section 4.3.7), and they serve as a defensible substitute for human-rater agreement when validating LLM-extracted features (Section 4.3.4).

The composite `sentiment_score` is retained alongside the channels rather than replaced by them. Three reasons motivate this. First, it preserves continuity with the Phase 1 headline bias experiment, which depends on a single sentiment value. Second, it allows ablation studies comparing the composite signal against the decomposed channels in the TFT (Section 4.3.7). Third, it provides a hedge against the channels turning out to be individually noisy at production scale.

### 3.3.5 Magnitude, event type, entities, certainty, and time horizon

The remaining fields in the Haiku schema capture orthogonal dimensions of news content. `magnitude` is a continuous score in [0, 1] measuring how market-moving an event is, with anchors provided in the prompt (0.0 = negligible event, 1.0 = historic; most articles score in the 0.1–0.4 range, with a major OPEC cut anchored at 0.9 and a full-scale geopolitical crisis at 1.0). `event_type` is an array of one to three categorical labels drawn from a fixed set of six values (`geopolitical`, `supply`, `demand`, `macro`, `technical`, `other`), ordered by salience within the article. `entities` is a list of named actors central to the article (countries, organisations, oil companies, key officials). `certainty` is a continuous score in [0, 1] distinguishing rumour, analyst forecast, and confirmed fact. `time_horizon` is a categorical label drawn from `immediate` (hours to a day), `short_term` (days to weeks), and `structural` (months or longer themes such as energy transition).

Each of these fields encodes information that FinBERT's three-class output cannot represent, and each enters the TFT v2 model either as a continuous input or as a categorical that the model's Variable Selection Network can weight independently. The design rationale for each field, including the choice to encode `event_type` as a list rather than a single value and the choice to handle entity normalisation in post-processing rather than in the prompt, is discussed in Section 4.3.2.

## 3.4 Filtering strategies

A substantial fraction of articles retrieved from GDELT is not substantive news. Common categories of non-substantive content include paywall placeholders, cookie consent notices, Cloudflare blocks, scraping error pages, generic stock-pick listicles, and articles whose body content is unrelated to oil markets despite matching one of the query keywords. Filtering these out is essential, since they would otherwise contribute noise to the sentiment signal and bias the downstream models.

We use two distinct filtering strategies across the two phases. Phase 1 used a regex/keyword heuristic on the article body. Phase 2 replaced this with an LLM-judged binary classification of content usability, exposed as the `usable` field in the Haiku extraction. Both approaches are retained as columns in the database (`body_valid` and `usable` respectively), enabling a direct comparison reported in Section 4.3.3.

### 3.4.1 Regex/keyword heuristic (Phase 1)

The Phase 1 filter, named `body_valid`, is an allow-list applied to the article body text. A body is considered valid if and only if it satisfies all of the following criteria:

- Minimum length of 400 characters (excluding whitespace and HTML residues)
- Presence of at least one term from a domain-specific keyword list covering energy, financial, and geopolitical vocabulary
- Absence of press-release boilerplate (vendor blacklist for known PR-distribution patterns)
- An all-caps ratio below 30%, computed over words of more than two characters in texts containing more than ten words (excluding scraped pages that are predominantly menu links, category headers, or stock-ticker listings)

The filter was developed iteratively over the early project, starting as a blacklist of known bad phrases (Cloudflare error text, cookie notices) and evolving into the allow-list form described above. On the Phase 1 corpus of 16,326 articles with English text and successful body retrieval, the filter classified 7,756 articles (47.5%) as having valid bodies. The remainder were either retained with title-only sentiment as a fallback or excluded from the Phase 1 analyses depending on the specific notebook (Section 4.2.1 details which analyses used which subset).

The regex heuristic is fast, reproducible, and transparent: any analyst can read the rules and predict their behaviour on a given article. It is also limited in ways that became evident during inspection. Articles passing the filter sometimes contained substantive boilerplate (long disclaimers, repeated company descriptions) that nevertheless mentioned energy keywords. Articles failing the filter sometimes contained legitimate analyst commentary whose body length fell just below the 400-character threshold, or whose vocabulary did not match the energy keyword list closely enough.

These observations motivated the migration to a more capable filter in Phase 2.

### 3.4.2 LLM-judged usable flag (Phase 2)

The Phase 2 filter, exposed as the `usable` boolean in the Haiku schema, asks the LLM itself to judge whether the article body is substantive WTI-relevant content. The prompt instructs the model to return `usable=false` for paywall placeholders, Cloudflare blocks, cookie notices, or articles that are clearly not about oil markets, macro, or geopolitics relevant to oil. When `usable=false`, the model is permitted (and instructed) to omit all other fields, exploiting the short-circuit optimisation discussed in Section 3.3.3.

The migration from `body_valid` to `usable` is motivated by three considerations. First, the LLM has access to semantic content that a regex cannot match: it can distinguish a substantive analyst piece from a press-release-shaped article that happens to contain energy keywords. Second, the LLM's judgement is calibrated against the same content model used for the rest of the feature extraction, so the filter and the features are coherent (an article flagged as usable will produce features that are themselves usable). Third, the marginal cost of sending more articles to the LLM under the Batches API and prompt-caching architecture (Section 3.3.3) is small relative to the methodological gain, so a broader filter at the input does not impose a meaningful resource constraint.

The `usable` flag becomes the canonical downstream filter in Phase 2 and beyond. All Phase 2 analyses condition on `usable=1` unless explicitly stated otherwise. The `body_valid` column is retained in the database for the comparative analysis reported in Section 4.3.3 but is no longer applied as a filter in any modelling step.

### 3.4.3 Comparative analysis

Both filter columns coexist in the database for the Phase 2 corpus, enabling a direct disagreement analysis. Of 19,619 Phase 2 articles processed by the LLM, 13,550 are flagged `body_valid=1` and 11,675 are flagged `usable=1`. The two filters disagree on a non-trivial proportion of articles in both directions: regex-rejected articles that the LLM accepted as substantive (the regex's false positives), and regex-accepted articles that the LLM rejected as boilerplate (the regex's false negatives). The full agreement analysis, including Cohen's kappa and worked examples of disagreements, is reported in Section 4.3.3.

## 3.5 Statistical models

Two classical time-series models support the empirical analysis of news impact on WTI liquidity. The lag-structured ordinary least squares (OLS) regressions, described in Section 3.5.1, provide the canonical statistical evidence for the research questions on lag structure (RQ1) and bearish-versus-bullish asymmetry (RQ2). The vector autoregression (VAR) model, described in Section 3.5.2, was implemented as a richer dynamic specification capable of representing bidirectional feedback between sentiment and liquidity. The VAR was ultimately abandoned for reasons of signal sparsity, documented in Section 4.2.5, but its specification is included here for completeness and because the abandonment itself motivates the move to the deep learning model of Section 3.6.

### 3.5.1 Lag OLS specification

The lag OLS analysis tests whether news sentiment published at hour `t` predicts trading volume at subsequent hours `t + k`, where `k` ranges over a fixed set of lag horizons. The specification at lag `k` is

```
log_volume_{t+k} = β_0 + β_1 · is_bearish_t + β_2 · is_bullish_t + ε_t
```

where `is_bearish_t` and `is_bullish_t` are binary indicators derived from the FinBERT categorical label for the article published at hour `t`. Neutral sentiment is the omitted reference category, so the coefficients `β_1` and `β_2` measure the expected difference in `log_volume` at hour `t + k` for bearish and bullish articles, respectively, relative to neutral articles. The use of two separate dummies rather than a single signed sentiment regressor is essential to the asymmetry question. A signed specification would force the model to assume that the bearish coefficient is the negative of the bullish coefficient (a symmetric response); the dummy specification allows the two coefficients to vary independently, which is precisely what RQ2 requires.

A separate regression is estimated at each lag value `k ∈ {1, 2, 3, 4, 6, 8, 12}`. The lag set is dense at short horizons, where information propagation is expected to be detectable, and sparser at longer horizons, where the effect is expected to attenuate. At each lag, we report the two coefficients, their standard errors, p-values, and the regression R². The R² is expected to be small in absolute terms, since news sentiment is one input among many that influence hourly trading volume; the modelling goal is to detect a structured pattern of coefficients across lags rather than to maximise explanatory power. A statistically significant coefficient at the conventional `p < 0.05` threshold, repeated across multiple adjacent lags with a coherent magnitude pattern, is the criterion for claiming a detectable lag effect.

The contemporaneous specification (`k = 0`) is reported separately as the null reference. At hour `t`, the article has just been published, and any volume effect would require the news to act on the market within the same closing-hour bar. The expectation is that this regression is statistically insignificant; if so, that null result motivates the lagged analysis. Section 4.2.3 reports the contemporaneous result, and Section 4.2.4 reports the lag regressions.

All regressions are estimated by ordinary least squares with the default standard error formula. Each observation is a single article. The Phase 1 corpus of 13,690 articles produces approximately 13,000 per-lag regression rows after dropping articles whose target hour `t + k` falls outside the coverage window. The regression treats these as a cross-section of articles rather than as a sequential time series, so heteroskedasticity- and autocorrelation-consistent (HAC) standard error corrections are not applied. The assumption may slightly underestimate true uncertainty if articles published close in time share unobserved common shocks, but the structured pattern of coefficient significance across multiple lags is robust to this concern: even with moderately wider confidence intervals, the pattern of bearish coefficients peaking at lag +6 hours with bullish coefficients showing the same pattern at lower magnitude would survive.

## 3.5.2 Vector autoregression specification

The VAR specification jointly models four variables on an hourly grid:

- `log_volume`: the natural logarithm of hourly trading volume
- `sentiment_score`: the FinBERT categorical label mapped to `{-1, 0, +1}`, aggregated to hourly resolution as described below
- `log_return`: the hourly log return
- `price_range`: the hourly Parkinson range

For each variable `y_i,t` in this four-dimensional vector, the VAR(p) model is

```
y_i,t = c_i + Σ_{j=1}^4 Σ_{l=1}^p A_{i,j,l} · y_{j, t-l} + u_{i,t}
```

with the lag order `p` selected by the Akaike information criterion (AIC) over candidate lag orders from 1 to 24 hours, and `u_t` a vector of innovations assumed to be jointly normal. The matrix `A_{·,·,l}` captures the cross-variable dependencies at lag `l`, and the contemporaneous correlation structure is recovered from the covariance matrix of the residuals. Stationarity of all four variables is verified before estimation via Augmented Dickey-Fuller tests; all four pass with p-values close to zero, indicating no unit-root concerns.

The construction of the VAR input matrix is methodologically important. The full hourly trading series for the Phase 1 coverage window is used as the spine: every trading hour contributes one row. The `sentiment_score` value at each hour is the mean of FinBERT scores across articles whose publication time falls within two hours of that trading hour, i.e. articles with `assignment_gap < 2h`. Hours with no contemporaneous articles are assigned `sentiment_score = 0` by convention. This filling rule preserves the joint observation structure that the VAR estimator requires (the four variables must be observed at every time index), but it does so at a methodological cost discussed below.

After cleaning (removing the first hour of each trading session for missing `log_return` and dropping any rows with zero `log_volume`), the VAR is estimated on approximately 10,800 hourly observations.

The output of interest is the impulse response function (IRF), which traces the response of `log_volume` to a one-unit shock to `sentiment_score` over a horizon of 24 hours. The IRF is the natural VAR analogue of the lag OLS coefficients: both measure how the news signal at hour `t` predicts liquidity at subsequent hours, but the VAR additionally accounts for the autoregressive dynamics of `log_volume` itself and for any contemporaneous correlation between the four variables.

The sentiment series in the VAR input is dominated by zeros. The mean and standard deviation of `sentiment_score` across the 10,800 hourly observations are −0.080 and 0.537 respectively, but this masks a more informative breakdown: approximately half of all hours have no contemporaneous articles and contribute a value of exactly zero. The VAR estimator cannot distinguish "no news event occurred at this hour" from "the data is missing for this hour"; both produce a zero in the sentiment column. The consequence is that the VAR's estimated dynamics for `sentiment_score` are dominated by the zero-filled hours rather than by the hours where actual news occurred, and the resulting impulse response function carries wide error bands that span zero at most horizons. The detailed empirical findings, including the IRF figure and the reasoning that led to the move to the Temporal Fusion Transformer, are reported in Section 4.2.5.

Amihud illiquidity is not included in the VAR specification. Although Amihud is used as a liquidity feature in both the Phase 1 and Phase 2 TFT models (Section 3.6), it was not added to the four-variable VAR system. With hindsight, including Amihud would have been a reasonable choice, but the VAR's central limitation, the sparsity of the sentiment signal, would not have been alleviated by adding additional liquidity variables. The decision to move beyond the VAR rather than to extend it reflects a methodological judgement that the data structure called for a different model class, not a different variable list.

## 3.6 Temporal Fusion Transformer

The Temporal Fusion Transformer (TFT) [N], introduced by Lim, Arık, Loeff, and Pfister, is a deep-learning architecture for multi-horizon time-series forecasting with mixed continuous and categorical inputs. We adopt the TFT as the primary deep-learning model for the Phase 2 work, training two successive instances (TFT v1 and TFT v2) that share the same architectural skeleton but differ in their feature configuration and training data. This section describes the architecture itself; the feature configurations for each instance are documented in Sections 4.3.5 and 4.3.7.

### 3.6.1 Architecture overview

The TFT combines four architectural components into an end-to-end model:

- A **Variable Selection Network (VSN)** at each input time step, which assigns a learned scalar weight to each input feature, allowing the model to suppress uninformative features automatically
- A pair of **LSTM encoder–decoder layers** that process the history (encoder) and the forecast horizon (decoder) at fixed resolution
- An **interpretable multi-head self-attention** layer over the encoded sequence, exposing which historical time steps the model attends to when producing each forecast
- A **quantile output head** that produces multiple quantile predictions in a single forward pass, enabling prediction intervals without separate models

Each component contributes a specific interpretability benefit. The Variable Selection Network produces per-feature importance scores that are directly comparable across features, addressing the question of which inputs the model actually uses. The attention layer produces per-time-step weights that expose the temporal structure of the model's reasoning, which is directly relevant to RQ1 on the lag structure of news impact. The quantile output enables uncertainty quantification without separate model training.

### 3.6.2 Input typing

The TFT distinguishes three categories of inputs:

- **Time-varying known reals**: features whose values are known at every time step in both the history and the forecast horizon. In our setup these are calendar features (hour, day_of_week, month) and binary regime indicators (is_us_session, is_wednesday as a proxy for the EIA release day).

- **Time-varying unknown reals**: features whose values are known only over the history and must be predicted for the forecast horizon. These include the target itself (log_volume), the market-context features (price_range, log_return, amihud, dxy, vix), and the news-derived features extracted by Haiku (see Sections 4.3.5 and 4.3.7 for the per-instance feature lists).

- **Static categoricals**: features that are fixed for the duration of a time series. In our single-asset configuration this reduces to a single asset identifier (asset = 'WTI').

The Variable Selection Network learns separate gating weights for each of these categories, so a feature that is informative in the history (as an unknown real) but uninformative as a calendar input is correctly weighted across both contexts.

### 3.6.3 Forecast configuration

The TFT is configured to predict the target variable (`log_volume`) one hour ahead, conditioned on a fixed-length history window. The encoder length is set to 48 hours, providing two full daily cycles of context for each prediction. This encoder length is intentional: the lag OLS analysis in Phase 1 (Section 4.2.4) identified a news-impact peak at lag +6 hours, and an encoder window of 48 hours comfortably covers both this peak and the second-day "memory" effects observed in earlier TFT runs. The prediction length is set to one hour, matching the resolution of the target series.
The model is trained with a quantile loss [N], producing point predictions at multiple quantiles in a single forward pass. The quantile loss is robust to outliers in the target and provides calibrated uncertainty estimates without distributional assumptions.

### 3.6.4 Architectural hyperparameters

The TFT architecture exposes a small set of hyperparameters governing model capacity. We retain a single set of values across both TFT v1 and TFT v2 to enable direct comparison of the two instances on the same architectural footprint:

- `hidden_size = 32`: the internal representation dimension
- `attention_head_size = 4`: the number of parallel attention heads
  `dropout = 0.1`: regularization rate on internal layers
- `hidden_continuous_size = 16`: the embedding dimension for continuous variable processing
- `learning_rate = 1e-3`: the initial Adam optimizer learning rate, with on-plateau reduction (patience 3)

These values yield a model with approximately 113,000 trainable parameters. The choice to keep architectural hyperparameters fixed rather than tuning them separately for each instance reflects a methodological decision: the substantive difference between TFT v1 and TFT v2 is in the input features (the Haiku schema iteration and the corpus extension), not in model capacity, and isolating the feature-level differences requires holding architecture constant.

### 3.6.5 Training procedure

Training is performed in PyTorch Lightning [N] via the `pytorch-forecasting` library [N]. The training loop uses:

- Adam optimization with the learning rate above
- Early stopping on validation loss with a minimum improvement of `1e-4` and a patience of 5 epochs
- A maximum of 30 epochs (early stopping typically halts training earlier)
- Gradient clipping at 0.1 to stabilize updates with the small batch sizes characteristic of long-sequence forecasting.
- A Google Colab T4 GPU for TFT v1 training and an equivalent accelerator setup for TFT v2 (Section 4.3.7)

The dataset is split temporally, with the most recent 20% of hours used as the validation set and the earlier 80% used for training. Within the training set, fixed-length sliding-window samples are constructed by the `pytorch-forecasting` `TimeSeriesDataSet` wrapper, which handles the consistent encoding of categorical features, the construction of input tensors, and the application of group normalization to the target series. The validation set uses the same wrapper with `stop_randomization=True` to ensure deterministic evaluation.

### 3.6.6 Interpretation outputs

For each trained TFT, three categories of interpretability output are produced and reported in Chapter 4:

- **Validation loss** as the headline performance metric, allowing comparison between the v1 and v2 instances on the same temporal holdout
- **Variable Selection Network feature importance**, averaged over the validation set, providing a ranking of which input features the model relied on most heavily
- **Multi-head attention weights**, averaged over the validation set, providing a per-lag map of which historical hours the model attended to most heavily

The attention output is of particular methodological interest for RQ1. If the TFT attention pattern peaks at a lag consistent with the lag OLS regression result of Phase 1, the two analyses corroborate one another from very different methodological starting points: the OLS detects a structured lag effect from a sequence of single-lag linear specifications, and the TFT learns a single nonlinear function over the full input history while exposing the same structure through attention. Such corroboration from independent methods is a stronger combined finding than either analysis alone could produce.

## 3.7 Inter-model calibration methodology

The Haiku-extracted feature set described in Section 3.3 is the substrate of all Phase 2 modeling. Before relying on it for downstream analysis, we need an external check that the extraction is producing sensible outputs and that the LLM is not systematically biased on the cases that matter most. This section describes the methodology for that external validation. The specific calibration results, and the methodological refinement that the calibration motivated, are reported in Chapter 4 (Sections 4.3.4 and 4.3.5).

### 3.7.1 The validation problem and our approach

The standard approach to validating an NLP extraction is to compare model output against human-annotated ground truth on a labelled subset of the data. For this thesis, that approach is not feasible. Human annotation of WTI-relevant news articles requires domain expertise in oil markets and trading that the author does not possess, and a single non-expert annotator working at scale would introduce noise of unknown structure into the validation set rather than reduce it. Recruiting a third-party expert at scale was outside the resource scope of the project.

We adopt instead an inter-model calibration approach: the same articles, with the same extraction prompt and tool schema, are scored by Claude Haiku 4.5 and by a GPT-family reference model (the OpenAI GPT-4 series via the ChatGPT interface). Disagreement between two models from different developer families on the same input is informative in a different way than human disagreement, but is informative nonetheless. The argument for treating it as a useful validation signal is the following.

Each model encodes its own implicit notion of news interpretation, shaped by its training data, its alignment procedure, and the family-level inductive biases of its developer. The two models share little in their training data composition or alignment style. When they agree on an extraction, that agreement reflects content that is robust to two independently-trained interpretations of financial news. When they disagree, the disagreement points to either genuine ambiguity in the article or to a systematic bias in one of the models.

This approach is acknowledged as methodologically weaker than expert-annotated ground truth: two models agreeing on a wrong reading would not be detected by the comparison. The limitation is reported as part of the discussion in Chapter 5. It is, however, methodologically stronger than no external validation, and the cross-family choice maximises the diversity of the comparison within the constraint that both annotators are LLM-based.

### 3.7.2 Sample design

The calibration sample is constructed by stratified random sampling from the full Phase 2 corpus. The strata reflect the regex-based body validity classification used in Phase 1, with three bucket categories:

- **valid**: articles passing the regex `body_valid=1` filter (Section 3.4.1)
- **invalid_substantial**: articles failing `body_valid=1` but whose body field still contains substantial text content (length above a minimum threshold)
- **invalid_weak**: articles failing `body_valid=1` with bodies short enough to be likely scraping artefacts

From each of these three strata, ten articles are sampled uniformly at random, producing a total calibration sample of thirty articles. The stratification ensures that the calibration covers the full range of content the LLM might encounter in production, including the marginal cases where the regex filter and the LLM are most likely to disagree.

### 3.7.3 Extraction procedure

The same system prompt and tool schema (Appendix A) are submitted to both models for each of the thirty articles. For Claude Haiku, this is the production extraction code described in Section 3.3.2. For the GPT reference, the prompt and schema are submitted via the ChatGPT interface with manual JSON output extraction.

The two output sets are then aligned by article identifier and compared field by field. The comparison computes agreement metrics symmetrically: no metric depends on which model is treated as the reference.

### 3.7.4 Agreement metrics

For each field in the schema, we compute one or more of the following agreement measures, chosen to match the field's type:

- For binary fields (`usable`): agreement rate as a count out of thirty, and Cohen's kappa to correct for chance agreement
- For continuous fields (`sentiment_score`, `supply_impact`, `demand_impact`, `risk_premium`, `magnitude`, `certainty`): Pearson correlation across all pairs, mean absolute difference (MAD), and the count of **sign disagreements**, defined as cases where the two models' scores have opposite signs (one positive, one negative). Near-zero scores within a small range around zero are treated as agreement on neutrality and do not contribute to the sign-disagreement count.
- For categorical fields (`event_type`, `time_horizon`): exact-match agreement rate, and for the array-valued `event_type`, the proportion of cases where the top-ranked category from one model appears anywhere in the other model's list

The sign-disagreement count for continuous fields is the most methodologically important measure. A small mean absolute difference can mask cases where the two models disagree on the qualitative direction of an effect (one rates an event bullish, the other bearish), which is precisely the kind of disagreement that would propagate as noise into downstream models. The sign-disagreement count surfaces these cases explicitly.

### 3.7.5 Use of the calibration outcome

The calibration sample is small (n=30) by deliberate design: a larger sample would have proportionally larger annotation cost on the GPT side, and the goal is to detect qualitative patterns of agreement and disagreement rather than to produce a precise estimate of population-level agreement rates. The thirty articles are enough to surface systematic disagreements, identify their character, and motivate either acceptance of the schema as-is or schema revision.

In the event the calibration reveals significant disagreement on any field, the response is to investigate the source of the disagreement and revise the extraction prompt or schema accordingly, then re-run the calibration on the same thirty-article sample with the revised setup. This iterative loop is documented in Chapter 4 (Section 4.3.4 reports the first calibration result, Section 4.3.5 reports the schema revision, and the calibration is re-run with the revised schema in the same section). The use of the same thirty articles for both the initial and the revised calibration is intentional: it isolates the effect of the schema change from the effect of sample variation.
