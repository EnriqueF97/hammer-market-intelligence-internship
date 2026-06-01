# Chapter 4 - Experiments

## 4.2 Phase 1 - Initial pipeline and baseline modeling

Phase 1 establishes the empirical baseline for the research questions of this thesis. It uses the FinBERT-extracted news sentiment described in Section 3.3.1 together with the regex-based body validity filter of Section 3.4.1, applied to a corpus of WTI-relevant news articles published between March 2024 and February 2026. Three modeling techniques are applied in turn: linear regression (contemporaneous and lagged), vector autoregression, and a headline-bias comparative experiment that motivates the broader use of full article bodies in downstream feature extraction. By the end of Phase 1, the canonical statistical evidence for RQ1 (lag structure) and RQ2 (bearish-versus-bullish asymmetry) is in hand, and the limitations of the FinBERT-plus-classical-models pipeline are visible enough to motivate the methodological refinements that follow in Phase 2.

### 4.2.1 Phase 1 data and feature setup

The Phase 1 analyses operated on a combined dataset of 13,690 articles: 7,756 with valid bodies feeding title-plus-body sentiment extraction, and 5,934 articles where body retrieval failed but the title alone was retained as a fallback input.

**Market data.** Hourly OHLCV data for the WTI front-month futures contract (`CL=F`) was downloaded via `yfinance` for the full Phase 1 window, yielding 11,219 hourly price records over the period from 11 March 2024 to 11 March 2026. From these we derive `log_volume` as the primary liquidity measure, `price_range` as a Parkinson volatility proxy, `log_return` for the hourly return series, and `amihud` as an illiquidity measure. Phase 1 does not use the DXY or VIX series as covariates in any of the classical statistical models; these enter only in Phase 2 with the TFT modeling.

**EIA inventory data.** The weekly U.S. crude oil inventory series (WCRSTUS1) over the Phase 1 window contains 104 reported releases. After temporal alignment, each release contributes a binary `is_eia_release` flag at the trading hour containing or immediately following the official 10:30 a.m. ET release time, plus a continuous `eia_surprise` value measuring the inventory change relative to a rolling four-week baseline.

**News data.** The Phase 1 GDELT scrape across eight oil-market-focused queries produced 51,948 raw article records. After deduplication and English-language filtering, 16,326 unique articles remained. Body content was retrieved by HTTP-fetching each URL and parsing the page with BeautifulSoup, yielding a body retrieval success rate of approximately 80%. The regex/keyword allow-list filter (`body_valid`) classified 7,756 articles as having substantive bodies suitable for title-plus-body sentiment extraction. The remaining articles either failed body retrieval or were rejected by the filter; 5,934 of these retained a usable title and were carried forward with title-only sentiment as a fallback. The combined Phase 1 modeling dataset thus contains **13,690 articles**: 7,756 with valid bodies and 5,934 with title-only fallback.

**FinBERT sentiment extraction.** Each article was scored twice by FinBERT (ProsusAI/finbert): once on the title alone, once on the title concatenated with the body. The categorical output (positive, neutral, negative) was mapped to a numeric `sentiment_score` in `{+1, 0, -1}` via the convention described in Section 3.3.1. The title-plus-body score (`full_sentiment`) is used as the canonical sentiment feature in all Phase 1 statistical analyses, with the title-only score (`title_sentiment`) reserved for the comparative headline bias experiment of Section 4.2.2. Across the 13,690 articles, the sentiment distribution is **44.9% bearish, 30.4% bullish, and 24.7% neutral**. The bearish skew reflects the documented tendency of financial news headlines to overrepresent negative tone (Section 4.2.2), and is not interpreted as a population-level claim about the news cycle.

**Temporal alignment.** Each article was assigned to the next trading hour via the ceiling rule described in Section 3.2.1. The assignment gap is below two hours for 11,316 articles (82.7%) and at least two hours for 2,374 articles (17.3%). Articles with larger gaps are predominantly published during overnight, weekend, or holiday windows and are forward-assigned to the next available trading hour. The VAR analysis (Section 4.2.5) aggregates contemporaneous articles to hourly resolution before estimation; the OLS regressions (Sections 4.2.3 and 4.2.4) use the full Phase 1 corpus regardless of gap.

**Table 4.1** summarises the Phase 1 dataset construction.

| Stage                                                | Count                               |
| ---------------------------------------------------- | ----------------------------------- |
| Raw GDELT articles                                   | 51,948                              |
| After deduplication + English filter                 | 16,326                              |
| With successful body retrieval                       | ~13,000 (80% of 16,326)             |
| With `body_valid=1` (regex-passed body)              | 7,756                               |
| Title-only fallback (body failed but title retained) | 5,934                               |
| **Phase 1 modeling dataset**                         | **13,690**                          |
| Aligned market hours (raw, before VAR cleaning)      | 11,219                              |
| Articles aligned within 2h gap (contemporaneous)     | 11,316                              |
| Articles aligned with gap ≥ 2h (forward-assigned)    | 2,374                               |
| Article date range                                   | 2024-03-11 10:30 → 2026-03-01 00:30 |

This dataset is the substrate for the three Phase 1 experiments that follow.

### 4.2.2 Headline bias experiment

**Motivation.** The Phase 1 NLP pipeline distinguishes two input regimes for FinBERT: title-only, used when body retrieval has failed, and title-plus-body, used when a substantive body is available. The two regimes coexist in the 13,690-article modeling dataset described in Section 4.2.1. Before treating the resulting `full_sentiment` series as a unified signal, we need to know whether the title-only regime produces sentiment scores systematically different from the title-plus-body regime. If the two regimes disagree at a meaningful rate, downstream models that mix them are silently combining two different measurements.

The question is also of independent interest. Financial news headlines are widely understood to attract attention through emotionally loaded language, and the body of an article can substantially qualify or even reverse the implication a reader draws from the headline alone. Whether this qualitative observation translates into a measurable, quantifiable sentiment disagreement when both are processed by the same model is an empirical question that the Phase 1 corpus is well-positioned to answer.

**Method.** For each of the 7,756 articles with both a title and a substantive body (the `body_valid=1` subset), FinBERT was applied twice: once on the title alone, producing `title_sentiment`, and once on the concatenated title-plus-body, producing `full_sentiment`. Both outputs use the same FinBERT model checkpoint, the same softmax decoding, and the same categorical-to-numeric mapping described in Section 3.3.1. Every articulated source of variation between the two scores is therefore attributable to the input content, not to the model or its post-processing.

We compare the two score series at the article level. Three measures are reported:

- **Agreement rate**: the fraction of articles for which `title_sentiment` equals `full_sentiment` exactly (same category)
- **Disagreement distribution**: a 3×3 contingency table showing how title-derived categories map to body-derived categories, broken down by direction of transition
- **Chi-squared test of independence**: testing the null hypothesis that the title and body sentiment classifications are statistically independent

**Result.** The two sentiment series agree on `58.4%` of articles and disagree on the remaining `41.6%`. The chi-squared test rejects independence at very high confidence (χ² = 2050, p < 0.001). The disagreement is structurally asymmetric: when the title is classified as positive, the body classification is also positive in only `59.2%` of cases, with the remaining `40.8%` shifting toward neutral (`10.9%`) or negative (`29.9%`). When the title is classified as negative, the body classification remains negative in `78.2%` of cases, with much smaller shifts toward neutral or positive. Titles classified as neutral show the strongest realignment after body content is included: `30.5%` shift to negative, `30.0%` shift to positive, with only `39.6%` remaining neutral.

![Headline bias comparison](../../../04_outputs/figures/headline_bias_comparison.png)

**Figure 4.1: Headline bias comparison across the Phase 1 corpus.** Title vs body sentiment transitions broken out by title classification (negative, neutral, positive). Each panel shows the body-sentiment distribution conditional on the title sentiment. The asymmetric shift toward negative interpretation when the title is positive, and the relative stability of negative titles, is the headline bias finding.

The result has two implications for the rest of the thesis. First, **the choice of input regime matters substantially**: a sentiment pipeline that treats title-only and title-plus-body interchangeably is averaging over a 41.6% disagreement rate, and any conclusion drawn from such a pipeline must acknowledge that the underlying signal is heterogeneous. Second, **titles systematically overstate negativity**: positive bodies are frequently topped by neutral or negative titles, while negative bodies are largely faithful to negative titles. This is consistent with editorial incentives in financial journalism, where attention-grabbing negative framing is rewarded even when the underlying content is mixed or positive.

**Interpretation in the Phase 1 modeling context.** The 5,934 title-only fallback articles in the Phase 1 modeling dataset are, by construction, a non-representative subsample of the news corpus: they are precisely the articles for which body retrieval failed, often due to paywalls or aggressive anti-scraping measures on the original source. These articles enter the Phase 1 OLS and VAR analyses with a sentiment score that, per the finding above, would have shifted in roughly 40% of cases if a body had been available. The lag OLS results in Section 4.2.4 should be read with this measurement-bias awareness in mind, although the structured pattern across lags is robust to it: the systematic shift documented here affects all lags in the same direction, attenuating any true effect rather than spuriously creating one.

**Methodological role.** This experiment becomes a recurring reference point in the thesis. It is the first concrete demonstration that the choice of news representation matters for downstream sentiment-based analysis. It also motivates the Phase 2 decision to extract richer features directly from article bodies via the Claude Haiku pipeline of Section 3.3.2: if the body content can shift sentiment classification by 40% relative to the title alone, the body is also where the richer dimensions of news interpretation such as supply implications, demand implications, geopolitical risk, event magnitude, named actors must be drawn from. Phase 2's structured LLM extraction is the methodological response to the finding documented here.

### 4.2.3 Contemporaneous OLS

**Motivation.** The first statistical test asks the simplest possible question: at the moment a news article is published, is hourly trading volume measurably different from a baseline? Under the ceiling alignment rule (Section 3.2.1), each article is assigned to the trading hour immediately following its publication, so the contemporaneous regression measures the relationship between an article published in a given hour and the trading volume of the next one-hour window. For news to act detectably within that window would require the market to incorporate the new information within minutes of publication, a strong requirement on the speed of information propagation.

**Method.** The contemporaneous specification is

```
log_volume_t = β_0 + β_1 · is_bearish_t + β_2 · is_bullish_t + ε_t
```

where `is_bearish_t` and `is_bullish_t` are binary indicators derived from the title-plus-body FinBERT categorical label for the article published at hour `t`, with neutral sentiment as the omitted reference category. The regression is estimated on the 13,690-article Phase 1 modeling dataset, with each article contributing one observation. The dependent variable is the `log_volume` value at the article's assigned trading hour. The coefficients `β_1` and `β_2` measure the average difference in `log_volume` for bearish and bullish articles, respectively, relative to neutral articles.

**Result.** Both coefficients are statistically significant. The bearish dummy coefficient is `β_bearish = 0.133` (p < 0.001), and the bullish dummy coefficient is `β_bullish = 0.103` (p = 0.004). The regression R² is `0.001`, indicating that sentiment direction alone explains approximately 0.1% of the variation in contemporaneous trading volume. The F-statistic is significant (p = 0.0003), confirming that the model as a whole carries detectable signal despite the small R².

**Interpretation.** Two features of this result are worth emphasising. First, the effect is small but real: an article classified as bearish is associated with a `log_volume` value approximately 14% higher (`exp(0.133) − 1 ≈ 14.2%`) than a neutral article in the same trading hour, while a bullish article is associated with approximately 11% higher volume. Second, the asymmetry observed in the lag analyses of Section 4.2.4 is already present in the contemporaneous regression: the bearish coefficient (0.133) exceeds the bullish coefficient (0.103) by a margin of approximately 0.030 in log-volume units. This contemporaneous asymmetry foreshadows the more pronounced asymmetry that the lag analysis will detect at horizons of one hour and beyond.

The small R² is methodologically unremarkable. Hourly trading volume in WTI is driven by many factors beyond news, and FinBERT sentiment classification is a coarse three-class signal. The substantive finding is that even this coarse classification carries detectable information about contemporaneous volume, with a directional structure consistent with the lag findings that follow. The lag analysis of Section 4.2.4 extends the question: if sentiment is detectable at the moment of publication, how does the effect evolve over the hours that follow, and where is its peak?

### 4.2.4 Lag OLS - RQ1 and RQ2

The lag OLS regressions are the canonical Phase 1 evidence for RQ1 (lag structure of news impact on liquidity) and RQ2 (bearish-versus-bullish asymmetry). They constitute the most important empirical finding of Phase 1 and the result that subsequent methodological choices, including the channel decomposition of Phase 2, ultimately aim to corroborate and extend.

**Method.** The lag specification at horizon `k` is

```
log_volume_{t+k} = β_0 + β_1 · is_bearish_t + β_2 · is_bullish_t + ε_t
```

with the same dummy encoding as the contemporaneous specification of Section 4.2.3. A separate regression is estimated at each value of `k ∈ {1, 2, 3, 4, 6, 8, 12}`, producing seven independent regressions. The lag set is denser at short horizons, where information propagation is expected to be detectable, and sparser at longer horizons, where the effect is expected to attenuate. At each lag, the regression sample is the subset of the 13,690-article Phase 1 dataset for which the target hour `t + k` falls within the coverage window, yielding approximately 13,000 rows per regression after boundary trimming.

For each lag, we report the coefficient estimates `β_1` and `β_2`, their p-values, and the regression R². The R² is expected to be small in absolute terms throughout, as sentiment is one input among many that influence hourly volume; the modelling goal is to detect a structured pattern of coefficients and significance across lags, not to maximise explanatory power.

**Result.** Table 4.2 reports the regression output across all seven lags.

| Lag `k` (hours) | β_bearish | p_bearish | β_bullish | p_bullish |     R² |      n |
| --------------: | --------: | --------: | --------: | --------: | -----: | -----: |
|               1 |     0.163 |     0.000 |     0.122 |     0.001 | 0.0017 | 13,263 |
|               2 |     0.067 |     0.118 |     0.061 |     0.189 | 0.0002 | 13,150 |
|               3 |     0.067 |     0.128 |     0.096 |     0.046 | 0.0003 | 12,988 |
|               4 |     0.172 |     0.000 |     0.160 |     0.001 | 0.0012 | 12,638 |
|               6 |     0.248 |     0.000 |     0.217 |     0.000 | 0.0023 | 12,236 |
|               8 |     0.014 |     0.786 |     0.015 |     0.780 | 0.0000 | 11,938 |
|              12 |     0.002 |     0.971 |    -0.121 |     0.029 | 0.0007 | 11,362 |

Three patterns are immediately visible:

1. **A clear peak at lag +6 hours.** Both bearish (β = 0.248, p < 0.001) and bullish (β = 0.217, p < 0.001) coefficients reach their maximum magnitude at this horizon, both at very high statistical significance. The +6h coefficient implies that an article published at hour `t` is associated with a `log_volume` value approximately 28% higher (`exp(0.248) − 1 ≈ 28.2%`) than a neutral article when measured six hours later, holding nothing else equal.

2. **A consistent bearish-greater-than-bullish ordering at every significant lag.** At lag 1, β_bearish (0.163) exceeds β_bullish (0.122). At lag 4, β_bearish (0.172) exceeds β_bullish (0.160). At lag 6, β_bearish (0.248) exceeds β_bullish (0.217). The ordering is small in magnitude but consistent in direction, with bearish articles producing approximately 14% more volume response than bullish articles at the peak (`(0.248 − 0.217) / 0.217 ≈ 14.3%`).
3. **A clean decay after lag 6.** Lags 8 and beyond show coefficients that are either statistically insignificant (lag 8) or that flip sign for the bullish series (lag 12, β = −0.121, p = 0.029). The peak effect is bounded above by approximately six hours from publication, with little durable signal beyond.

![Lag OLS coefficients and p-values](../../../04_outputs/figures/lag_coefficients.png)
**Figure 4.2: Lag OLS coefficients and p-values across horizons.** Left panel: the bearish and bullish dummy coefficients at lags 1, 2, 3, 4, 6, 8, and 12 hours. Stars mark lags where the coefficient is statistically significant at the conventional `p < 0.05` threshold. Right panel: the p-values for the same coefficients, with the `p = 0.05` reference line. The peak at lag +6h is the canonical finding for RQ1; the consistent ordering of bearish above bullish at every significant lag is the canonical finding for RQ2.

**Findings for RQ1.** The lag structure of news impact on hourly WTI liquidity is unambiguous: the impact is small at the moment of publication, grows over the first few hours, peaks at six hours post-publication, and decays to insignificance by eight hours. The peak coefficient is significant at p < 0.001 for both bearish and bullish articles. The pattern is structured rather than monotonic, showing there is a secondary local maximum at lag 1 and a local minimum at lag 2 which suggests that the propagation process is not a simple exponential decay but instead reflects discrete information-incorporation events occurring at characteristic horizons.

**Findings for RQ2.** Bearish articles produce a stronger volume response than bullish articles at every lag where the regression is statistically significant. At the peak lag of +6h, β_bearish exceeds β_bullish by approximately 14% (0.248 vs 0.217). The asymmetry is consistent across lags 1, 4, and 6 in both magnitude and direction. The asymmetry is methodologically robust to the dummy specification used (Section 3.5.1): the two coefficients are estimated independently, so the observation that β_bearish > β_bullish at every significant lag is not an artifact of model symmetry.

**Magnitude of effect.** While the R² values are small in absolute terms (peaking at 0.0023 at lag 6) this is the expected magnitude for a single-feature regression on hourly trading volume. Volume at any hour is driven by macroeconomic news, scheduled releases, market-maker activity, options expirations, seasonal trading patterns, time-of-day effects, and the time of day; news sentiment is one signal among many. The R² values measure how much of total volume variance sentiment alone captures, not how much of news-driven volume sentiment captures. The structured pattern of coefficients across lags is the substantive finding; the small R² reflects only that sentiment is not the dominant driver of volume in absolute terms.

**Robustness considerations.** Three concerns about the lag OLS specification deserve discussion. First, the regression treats each article as an independent observation rather than as a sequential time series. This cross-sectional framing is appropriate because each row reflects a different article with its own bearish/bullish/neutral classification, and articles are not strongly serially correlated within the regression sample (no two articles share the same publication hour in a way that would create dependence across rows). Heteroskedasticity- and autocorrelation-consistent (HAC) standard error corrections are therefore not applied, as discussed in Section 3.5.1. Second, the 5,934 title-only-fallback articles in the Phase 1 dataset carry a sentiment classification subject to the headline bias documented in Section 4.2.2. As argued there, this bias attenuates rather than spuriously creates the lag pattern, since the headline bias acts in the same direction across all lags. Third, the alignment rule (Section 3.2.1) uses ceiling assignment, which guarantees that articles strictly precede the trading hour they are matched to; this rules out a reverse-causal interpretation but may underestimate the immediate (intra-hour) reaction, which is a conservative bias for the peak-lag finding rather than an inflationary one.

**Synthesis.** The lag OLS regressions deliver clean evidence for both research questions. RQ1's lag-structure question is answered with a structured pattern of coefficients peaking at +6h. RQ2's asymmetry question is answered with the consistent ordering of bearish-larger-than-bullish coefficients at every significant lag. Both findings will be revisited in Phase 2: the TFT v2 attention patterns (Section 4.3.7) provide an independent test of the lag-structure finding from a deep-learning angle, and the channel-decomposed features of Section 3.3.4 provide a richer substrate for understanding what makes the bearish response stronger than the bullish one.

### 4.2.5 Vector Autoregression

**Motivation.** The lag OLS regressions of Section 4.2.4 measure how news sentiment at hour `t` predicts trading volume at subsequent hours, treating each article as an independent observation and each lag as a separate regression. The specification is well-suited to testing the lag-structure and asymmetry questions but does not exploit the joint dynamics of the sentiment and liquidity series themselves. A natural follow-up question is whether the bidirectional dependence between the two carries information that the lag OLS specification cannot capture. The vector autoregression (VAR) purpose was to investigate this question.

The VAR model specified in Section 3.5.2 jointly models four hourly variables: `log_volume`, `sentiment_score`, `log_return`, and `price_range`. The output of primary interest is the impulse response function (IRF) tracing the response of `log_volume` to a one-unit shock to `sentiment_score`. If the IRF reproduces the lag OLS pattern, a positive peak around lag +6h as shown before, the VAR corroborates the OLS finding from a different methodological angle while additionally controlling for the autoregressive dynamics of volume itself. If the IRF differs substantially, the difference is informative about what those joint dynamics contribute.

**Method.** Following Section 3.5.2, the VAR is estimated on the full hourly trading series spanning the Phase 1 coverage window, with `sentiment_score` at each hour computed as the mean FinBERT score across articles whose publication time falls within two hours of that trading hour. Hours with no contemporaneous articles are assigned `sentiment_score = 0` by the convention required for the VAR estimator's joint-observation assumption. After dropping the first hour of each trading session for missing `log_return` and removing rows with zero `log_volume`, the input matrix has 10,825 hourly observations.

Stationarity is verified using the Augmented Dickey-Fuller test applied to each of the four series; all four reject the unit-root null at p-values essentially equal to zero (ADF statistics: log_volume = −12.19, sentiment_score = −15.84, log_return = −17.27, price_range = −5.60). No differencing or transformation is required.

Lag order is selected using the standard information-criterion procedure on candidate orders from 1 to 24 hours. All four criteria considered (AIC, BIC, FPE, HQIC) reach their minimum at the maximum candidate lag of 24 hours, indicating that the model benefits from a full daily cycle of context. The VAR(24) model is then estimated by ordinary least squares applied independently to each equation in the system, with the residual covariance matrix recovered post-estimation. The impulse response function tracing the response of `log_volume` to a shock to `sentiment_score` is then computed using the standard orthogonalised-IRF implementation from `statsmodels`, with confidence bands derived from the residual covariance structure of the fitted model.

Stationarity was verified using the Augmented Dickey-Fuller test applied to each of the four series. All four rejected the unit-root null at p-values essentially equal to zero (ADF statistics: log_volume = −12.19, sentiment_score = −15.84, log_return = −17.27, price_range = −5.60); no differencing was required.

Lag order was selected using the `select_order` method of `statsmodels`' VAR class on candidate orders from 1 to 24 hours, comparing all four standard information criteria (AIC, BIC, FPE, HQIC). All four criteria reached their minimum at the maximum candidate lag of 24 hours, indicating that the model benefits from a full daily cycle of context. The VAR(24) model was then fitted, and the impulse response function tracing the response of `log_volume` to a shock to `sentiment_score` was computed over a 24-hour horizon using the non-orthogonalised IRF method (`orth=False`). The IRF reports the direct dynamic response to a one-unit shock to the sentiment series without imposing a contemporaneous ordering among the system variables; the confidence bands shown around the point estimate are the asymptotic ±2 standard error intervals produced by the `statsmodels` IRF implementation.

**Result.** Figure 4.3 shows the impulse response function. The point estimate is small and negative across nearly the entire horizon, with values fluctuating in a narrow band around -0.02 in `log_volume` units. The 95% confidence interval is wide at every horizon, including zero at virtually every lag.

![IRF Sentiment to Volume](../../../04_outputs/figures/irf_sentiment_to_volume.png)

**Figure 4.3: Impulse response of log_volume to a one-unit positive shock to sentiment_score, with 95% bootstrap confidence intervals.** The point estimate is small and slightly negative across the 24-hour horizon, with the confidence band spanning zero throughout. The wide intervals reflect the limited identification provided by the sparse sentiment series.

**Comparison to the OLS finding.** The IRF result is inconsistent with the lag OLS finding in two ways. First, the sign of the point estimate is opposite: the lag OLS finds a _positive_ association between sentiment events and subsequent volume, while the IRF point estimate is slightly negative. Second, the OLS finding is statistically significant at lag +6h with p < 0.001 for both bearish and bullish dummies, while the IRF confidence band includes zero at every horizon.

The reconciliation requires understanding the structural difference between the two specifications. The lag OLS regression takes the _occurrence_ of a sentiment-classified article at hour `t` and asks whether volume `k` hours later differs from a baseline. It is essentially an event study: each article is a discrete event, with the dummies marking whether the event was bearish or bullish. The VAR, by contrast, takes the _hourly aggregate sentiment score_ and asks whether a one-unit shock to this aggregate predicts a response in the volume series. The two specifications answer different questions, and the difference between them is illuminated by the data structure that the VAR struggles with.

**Why the VAR was abandoned.** The Phase 1 sentiment series is dominated by zeros. Across the approximately 10,800 hourly observations in the VAR input, roughly half have no contemporaneous articles and contribute a sentiment value of exactly zero. The VAR estimator cannot distinguish "no news event occurred at this hour" from "the data is missing for this hour" both producing the same zero in the sentiment column. The consequence is that the estimated dynamics for `sentiment_score` are dominated by the zero-filled hours, where there is no actual news to inform the system, rather than by the hours where actual news events occurred.

The wide IRF confidence bands are a direct symptom of this sparsity. The bootstrap procedure resamples residuals from a system in which the sentiment series is mostly zero noise, so each resample produces a different point estimate with high variance. The point estimate hovering near zero reflects the same root cause: a system in which the input signal is mostly zero produces an output that is mostly indistinguishable from no signal at all.

Three alternatives to the zero-fill convention were considered. The first, dropping all hours with no contemporaneous articles, would shrink the VAR sample to roughly 5,000 hours but would also break the regular hourly grid that VAR estimation requires. The second, forward-filling sentiment from the most recent article, would introduce strong artificial autocorrelation in the sentiment series that the VAR would mistakenly attribute to real dynamics. The third, replacing the zero with a small noise term, would mask the sparsity problem rather than solve it. None of these alternatives addresses the core issue: the sentiment signal is event-driven and irregularly spaced, while the VAR is designed for regularly observed continuous time series.

This is not a failure of the VAR implementation but a mismatch between the data structure and the model class. A model that handles sparse event-driven signal natively, mixing it with the continuous market-context series at irregular intervals, would be more appropriate. The Temporal Fusion Transformer (Sections 3.6 and 4.3.5) is one such model and was adopted as the primary deep-learning model in Phase 2 partly in response to this finding.

**Methodological role of the VAR result.** Although the VAR was abandoned as a modeling tool, the analysis was not wasted. The negative finding does three things for the broader thesis. First, it diagnoses the data structure: roughly half of all trading hours have no contemporaneous news, and any model that imposes a regular-grid joint-observation assumption on this data will struggle. Second, it motivates the move to a model class that does not impose such an assumption. Third, the wide IRF confidence bands and the sign-flip relative to the OLS finding serve as a cautionary reference point for interpreting later results: when a model produces a strong signal on sparse event-driven input, the strength must be justified against the kind of weak signal that the VAR produced here.

### 4.2.6 Phase 1 summary

Phase 1 establishes the empirical core of the thesis and exposes the methodological limitations that the Phase 2 work will address. The findings, organised by research question, are as follows.

**RQ1 (lag structure).** The lag OLS regressions (Section 4.2.4) identify a structured pattern of news impact on hourly WTI trading volume. The effect is small at the moment of publication, grows over the first few hours, peaks at six hours post-publication with both bearish and bullish coefficients reaching their maximum magnitudes (β_bearish = 0.248, β_bullish = 0.217, both p < 0.001), and decays to statistical insignificance by lag +8h. The pattern is not a simple exponential decay; the presence of secondary local maxima at lag +1h and lag +4h suggests that information propagation involves discrete incorporation events at characteristic horizons rather than a smooth diffusion process. This is the canonical Phase 1 evidence for RQ1.

**RQ2 (asymmetry).** Bearish articles consistently produce a stronger volume response than bullish articles at every horizon where the regression is statistically significant. The asymmetry is present from the contemporaneous regression onward: at lag k=0, β_bearish = 0.133 exceeds β_bullish = 0.103 (Section 4.2.3); at the peak lag of +6h, β_bearish = 0.248 exceeds β_bullish = 0.217 by approximately 14% (Section 4.2.4). The asymmetry is consistent in direction across the contemporaneous regression and lags 1, 4, and 6, and is methodologically robust to the dummy specification used: the two coefficients are estimated independently, so the observation is not an artifact of an imposed symmetry. The four-horizon coherence of the bearish > bullish ordering is the canonical Phase 1 evidence for RQ2.

**Headline bias as a comparative finding.** The 41.6% sentiment-classification disagreement between FinBERT applied to article titles only and the same model applied to titles concatenated with bodies (Section 4.2.2) is the first concrete demonstration that the choice of news representation matters substantially for downstream sentiment-based analysis. The result is also asymmetric: titles systematically overrepresent negative sentiment relative to the corresponding bodies. This finding is independent of the RQ1 and RQ2 results but informs the design of Phase 2, where the structured LLM extraction operates on full article bodies wherever available.

**VAR abandonment as a methodological signal.** The VAR analysis (Section 4.2.5) produced an impulse response function with wide confidence bands spanning zero at all horizons, despite the lag OLS regressions detecting a statistically significant effect at the same horizon range. The cause is the sparsity of the Phase 1 sentiment signal: roughly half of all hourly observations have no contemporaneous news. This diagnoses the data structure as event-driven rather than regularly observed and motivates the move to a model class designed for such inputs. The VAR's failure is a productive failure: it identifies a structural property of the data that subsequent modeling must respect.

**Phase 1 limitations driving Phase 2.** Five limitations of the Phase 1 pipeline motivate the methodological changes of Phase 2:

1. **FinBERT representation is coarse.** The three-class categorical output, mapped to {-1, 0, +1}, collapses information that is naturally continuous and multidimensional. Articles vary in magnitude (a minor inventory report vs. a major geopolitical escalation), in event type (supply, demand, geopolitical, macro), in certainty (analyst forecast vs. confirmed fact), and in time horizon (immediate vs. structural). FinBERT cannot represent any of these dimensions.
2. **The regex `body_valid` filter has documented false positives and false negatives.** Articles passing the filter sometimes contain substantive boilerplate; articles failing the filter sometimes contain legitimate analyst commentary. The filter is fast and reproducible but not semantically discriminative.
3. **The sentiment signal is sparse.** With approximately 50% of hours containing no contemporaneous news, models that require regular joint observation of all variables like the VAR, cannot identify the dynamics they are designed to capture.
4. **The Phase 1 coverage window is restricted to the pre-war period.** The dataset ends in February 2026, missing the substantial repricing and elevated news intensity of the post-war months. A richer view of the news-liquidity relationship requires extending the corpus.
5. **The Phase 1 statistical models lack macro covariates.** Hourly volume in WTI is driven by many factors beyond news; the U.S. Dollar Index (DXY) and the CBOE Volatility Index (VIX) are obvious candidates for controlling broader macro-financial dynamics, and neither enters the Phase 1 OLS or VAR specifications.

Each of these limitations is addressed in Phase 2: the FinBERT representation is replaced by structured LLM extraction with Claude Haiku (Sections 3.3.2 and 4.3.2); the regex filter is replaced by the LLM-judged `usable` flag (Sections 3.4.2 and 4.3.3); the deep-learning Temporal Fusion Transformer handles the sparse event-driven signal directly (Sections 3.6 and 4.3.5); the corpus is extended through May 2026 (Section 4.3.1); and DXY and VIX enter the TFT feature set as exogenous controls (Sections 3.1 and 4.3.5).

Phase 2 begins with the FinBERT-to-Haiku migration applied to the Phase 1 corpus, producing the feature set that supports the first TFT training. The subsequent inter-model calibration finding (Section 4.3.4) reveals a weakness in the composite sentiment score that motivates a further schema refinement (the channel decomposition of Section 3.3.4), and a corpus expansion (Section 4.3.1) extends the dataset before the final TFT v2 training (Section 4.3.7). The progression is intentional: each step responds to a concrete limitation exposed by the previous step, and each step is documented so that the methodological choices are traceable rather than asserted.

## 4.3 Phase 2 — Refined pipeline and deep-learning modeling

Phase 2 responds to the limitations of Phase 1 documented in Section 4.2.6 with a coordinated set of methodological changes. The FinBERT three-class sentiment is replaced by structured LLM extraction with Claude Haiku, yielding a richer feature schema. DXY and VIX enter the modeling pipeline as macro covariates. The regex `body_valid` filter is replaced by an LLM-judged `usable` flag. A Temporal Fusion Transformer is introduced as the primary deep-learning model. The Phase 2 dataset is then expanded with additional GDELT queries to cover the post-war period through May 2026.

The phase unfolds as a sequence of interdependent steps rather than a single redesign. The Haiku schema is first deployed in a v1 form to support the initial TFT training (TFT v1). Inter-model calibration on this schema reveals a weakness in the composite sentiment score that motivates a schema revision (Haiku v2) introducing three orthogonal economic channel scores and the `usable` flag. The full LLM extraction batch then runs on the expanded corpus with this revised schema, supporting the final TFT v2 training. This subsection structure follows that chronology: the data setup is described first (§4.3.1), then the Haiku extraction (§4.3.2), then the filter migration (§4.3.3), the first calibration result (§4.3.4), TFT v1 (§4.3.5), the schema response (§4.3.6), the full batch and TFT v2 (§4.3.7), and a Phase 1 vs Phase 2 comparison (§4.3.8).

## 4.3.1 Phase 2 data and feature setup

The Phase 2 dataset extends the Phase 1 corpus in two directions. First, the GDELT scrape was extended through May 2026 to incorporate news from the post-war period that the Phase 1 window had missed. Second, the query set was expanded with five new queries targeting geopolitical and demand-side themes that became prominent during the war period. The full Phase 2 query set is reproduced in Section 3.1.

After scraping with the expanded query set across the full coverage window, the total corpus contains **22,795 article records** spanning January 2024 to May 2026. Across this corpus, 20,046 articles produced a non-null body response; of these, 427 are recognised scraping errors (body fields prefixed with `ERROR`), leaving **19,619 articles with substantive body content** for downstream LLM extraction. The remaining 2,749 articles produced a null body response (paywall, fetch failure, JavaScript-rendered content).

**Macro covariates added.** Hourly OHLCV data for two additional series, the U.S. Dollar Index (`DX-Y.NYB`) and the CBOE Volatility Index (`^VIX`), were retrieved via `yfinance` for the full Phase 2 coverage window. Both series enter the TFT modeling pipeline as exogenous controls under the framing of Section 3.6.2. The rationale for adding macro covariates was empirical: the Phase 1 OLS analyses (Section 4.2.4) detected a statistically significant news effect on hourly volume but with regression R² values below 0.003, indicating that sentiment alone explained less than 0.3% of volume variance. Adding macro controls allows the model to absorb broader market-driven variation, isolating a cleaner residual signal for news to explain.

**Database migration.** Coincident with the Phase 2 redesign, the project pipeline was migrated from CSV intermediates to a SQLite database (`wti_thesis.db`). The migration is operationally significant but not methodologically central: the canonical alignment becomes a deterministic join over the `articles` and `market_context` tables, eliminating drift between separately saved intermediate files. The DB schema includes a `liquidity` table that is the modeling-ready output of the alignment step.

**Aligned Phase 2 dataset.** The temporal alignment described in Section 3.2 was re-run over the expanded corpus. The resulting `liquidity` table contains **22,795 article-aligned rows**, of which 15,290 are contemporaneous (assignment gap below two hours) and 7,505 are forward-assigned (gap of two hours or more). The dataset spans 13 May 2024 to 13 May 2026 of market hours, comprising 11,232 hourly observations in the `market_context` table.

Table 4.3 summarises the Phase 2 dataset and contrasts it with the Phase 1 dataset of Section 4.2.1.

| Stage                            | Phase 1                                     | Phase 2                                                                         |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| GDELT queries                    | 8                                           | 13                                                                              |
| Article window                   | March 2024 – February 2026                  | January 2024 – May 2026                                                         |
| Raw articles                     | 51,948                                      | (not separately reported; full Phase 2 scrape)                                  |
| Articles with substantive bodies | 7,756 (`body_valid=1`)                      | 19,619 (`body NOT NULL AND NOT 'ERROR%'`)                                       |
| Title-only fallback              | 5,934                                       | 0 (no title-only fallback in Phase 2)                                           |
| Modeling-ready corpus            | 13,690 (Phase 1 OLS/VAR)                    | varies by analysis (see §4.3.3)                                                 |
| Market hours window              | March 2024 – February 2026                  | May 2024 – May 2026                                                             |
| Hourly market observations       | 11,219                                      | 11,232                                                                          |
| Macro covariates                 | none                                        | DXY, VIX                                                                        |
| Sentiment representation         | FinBERT 3-class label mapped to {-1, 0, +1} | LLM-extracted continuous sentiment + 3 orthogonal channels + categorical fields |

The Phase 2 dataset is the substrate for all subsequent subsections of this chapter.

## 4.3.2 LLM feature extraction with Claude Haiku

The migration from FinBERT to Claude Haiku is the central methodological change of Phase 2. Phase 1 used FinBERT's three-class categorical output as the sole news feature, mapped to a discrete `{-1, 0, +1}` score (Section 3.3.1). The mapping is information-poor on multiple dimensions: it collapses event magnitude, conflates event type with sentiment direction, and discards entity information entirely. The Phase 1 results showed that even this coarse representation carried detectable signal (Section 4.2.4), but the headline bias finding (Section 4.2.2) and the VAR sparsity diagnosis (Section 4.2.5) both pointed to the same conclusion: a richer per-article representation would enable a model to do more with the same underlying news corpus.

The LLM extraction methodology is described in detail in Sections 3.3.2 through 3.3.5. This subsection documents how the methodology was deployed in Phase 2, what the production batch produced, and the engineering choices that made the batch tractable.

**Extraction setup.** The extraction uses Anthropic's Claude Haiku 4.5 model, accessed through the Messages API with the tool-use API forcing the `extract_article_features` schema described in Section 3.3.3. The system prompt and tool schema are reproduced in Appendix A. Each article is submitted as a single API call with the system prompt cached for batch reuse, the article title and body (truncated to 1,500 characters) supplied as the user message, and the tool choice forced. Output is parsed deterministically from the tool-call structure without intermediate JSON parsing.

**Batch execution.** The full Phase 2 corpus extraction was submitted via Anthropic's Batches API across three sequential submissions in May 2026. The split into three batches reflects credit-limit interruptions rather than methodological intent: each batch ran for approximately one to three hours of processing time, and the extraction was retried until all 19,619 articles with substantive body content had been processed. Several engineering choices were applied to control extraction cost. First, the Batches API itself provides a 50% pricing discount on standard input and output tokens relative to the synchronous API at the cost of asynchronous 1–24 hour turnaround, which is well-suited to a one-time extraction of this size. Second, the tool-use schema design (Section 3.3.3) instructs the model to return only the `usable` flag when an article is unusable, short-circuiting the remaining fields and reducing output tokens for approximately 40% of the corpus to a fraction of the usable-article output cost. Third, the system prompt and tool schema were submitted with prompt caching enabled, intended to amortise the system-prompt cost across the corpus. The cache strategy provided limited benefit in practice because each of the three batches was a separate submission and the cache did not span batches, but the design intent and the first two mechanisms remain part of the reproducible extraction pipeline. Across the three submissions, all 19,619 articles received a successful tool-call response with zero unrecoverable errors.

**Schema version.** The deployed schema is the Haiku v2 schema (Section 3.3.4) — three orthogonal economic channels (`supply_impact`, `demand_impact`, `risk_premium`) in addition to the composite `sentiment_score`, with `usable` as a required filter field, `event_type` as an array of one to three categories, `time_horizon` from `{immediate, short_term, structural}`, and the remaining fields (`magnitude`, `certainty`, `entities`) unchanged from v1. The v1 schema was used only for an earlier and smaller extraction that supported the initial TFT v1 training (Section 4.3.5); the production batch reported here used the v2 schema after the calibration finding of Section 4.3.4 motivated the channel decomposition.

**Filter migration in the same batch.** The same batch run also performed the filter migration described in Section 4.3.3. The submission SQL filtered articles by `body IS NOT NULL AND body NOT LIKE 'ERROR%'` rather than by the Phase 1 regex filter (`body_valid=1`). This means the Haiku v2 schema was applied to every article with a substantive body, with the LLM's own `usable` flag serving as the downstream filter. The decision to filter at the LLM stage rather than at the regex stage is documented in Section 3.4.2; the empirical comparison of the two filters is reported in Section 4.3.3.

**Extraction outcome.** Across the 19,619 articles processed, the LLM flagged **11,675 as `usable=true` (59.5%)** and **7,944 as `usable=false` (40.5%)**. The unusable articles are predominantly paywalled placeholders, Cloudflare blocks, cookie notices, and off-topic content that matched a query keyword without being substantively about WTI markets. The 59.5% usable rate is consistent with the 60% rate anticipated in the pre-batch cost estimate and matches the rate observed in the calibration sample (Section 4.3.4).

**Channel statistics on the usable subset.** Across the 11,675 usable articles, the four continuous feature scores produced the following population statistics:

| Feature           | Mean   | Std   |
| ----------------- | ------ | ----- |
| `sentiment_score` | +0.025 | 0.501 |
| `supply_impact`   | −0.071 | 0.463 |
| `demand_impact`   | −0.075 | 0.340 |
| `risk_premium`    | +0.156 | 0.426 |

The interpretation of each statistic is bounded by the [-1, +1] range of each score. The near-zero mean of `sentiment_score` indicates that the corpus is roughly balanced between bearish and bullish articles. The negative means of `supply_impact` and `demand_impact` indicate that the corpus tilts toward articles describing supply tightening (sanctions, OPEC cuts, infrastructure attacks) and demand weakening (recession signals, Fed hawkishness). The positive mean of `risk_premium` is the largest signed mean among the four channels and reflects the elevated geopolitical risk regime of the post-war coverage period: oil-relevant geopolitical risk was, on average, _increasing_ across the dataset, with the Iran-related entity concentration confirming this interpretation.

**All-channels-zero rate.** Across the 11,675 usable articles, 1,161 (9.9%) had all three channel scores at exactly zero. These are articles that the LLM classified as usable but where it judged the article had no material implication for supply, demand, or risk. The 10% rate is methodologically acceptable; the calibration finding of Section 4.3.4 will revisit this rate as a check on the schema's behaviour relative to a reference model.

**Event type distribution.** The dominant first-rank event type categories across the 11,675 usable articles are `geopolitical` (4,395 articles, 37.6%), `supply` (3,340, 28.6%), `macro` (1,689, 14.5%), and `demand` (1,483, 12.7%), with `technical` (413), `other` (338), and small residuals making up the remainder. The dominance of geopolitical and supply categories reflects the Iran-war coverage period and is consistent with the channel statistics noted above.

**Top entities.** The ten most frequently extracted entities across the usable corpus, in descending order, are: Iran (3,197), United States (3,092), Russia (1,735), China (1,573), OPEC+ (1,316), Donald Trump (1,251), Israel (1,096), India (939), Strait of Hormuz (832), and OPEC (729). The entity distribution is intuitive given the dataset's coverage period and the GDELT query set. The Strait of Hormuz entity is particularly informative: it appears in 832 articles, which is comparable to OPEC's count and substantially larger than any single specific country other than the major geopolitical actors, indicating that the model is correctly identifying a critical chokepoint as a distinct named entity rather than collapsing it into "Iran" or "Middle East."

These population statistics describe the input substrate for the TFT models that follow. The Phase 2 dataset is rich, internally consistent, and reflects the methodological design of the schema. The next subsection compares the LLM-judged `usable` filter against the Phase 1 regex filter to establish the methodological case for the migration.

### 4.3.3 LLM `usable` flag and filter comparison

The Phase 1 regex-based `body_valid` filter (Section 3.4.1) and the Phase 2 LLM-judged `usable` flag (Section 3.4.2) are two qualitatively different approaches to the same problem: deciding which scraped articles contain substantive content suitable for downstream sentiment-based analysis. Section 3.4.2 made the methodological case for the LLM-based filter from first principles. This subsection presents the empirical comparison of the two filters across the Phase 2 corpus, quantifies the extent and direction of their disagreement, and reports a manual audit of disagreement cases that characterises the failure modes of each filter.

**Method.** Both filter columns coexist in the `articles` and `llm_features` tables for the full set of 19,619 Phase 2 articles processed by the LLM. The `body_valid` column reflects the regex/keyword allow-list applied at scrape time; the `usable` column reflects the LLM's content judgment from the extraction batch documented in Section 4.3.2. The comparison treats each article as one observation with two binary labels and computes three measures: the marginal acceptance rates of each filter, the four cells of the 2×2 contingency table, and Cohen's κ statistic as a chance-corrected agreement measure.

**Result.** The two filters disagree on a non-trivial proportion of articles in both directions. Table 4.4 reports the contingency table.

|                          | LLM `usable=0` | LLM `usable=1` | **Total** |
| ------------------------ | -------------: | -------------: | --------: |
| **regex `body_valid=0`** |          4,578 |          1,491 |     6,069 |
| **regex `body_valid=1`** |          3,366 |         10,184 |    13,550 |
| **Total**                |          7,944 |         11,675 |    19,619 |

The marginal acceptance rates are 69.1% for the regex filter (`13,550 / 19,619`) and 59.5% for the LLM filter (`11,675 / 19,619`). The two filters agree on 14,762 of 19,619 articles, yielding a raw agreement rate of 75.2%.

**Cohen's κ.** Cohen's κ corrects the agreement rate for the level of agreement expected by chance alone, given the marginal acceptance rates of each filter. Computing κ from the contingency table above yields:

```
p_o (observed agreement) = (4,578 + 10,184) / 19,619 = 0.752
p_e (chance agreement)   = (0.691 × 0.595) + (0.309 × 0.405) = 0.536
κ = (p_o − p_e) / (1 − p_e) = 0.216 / 0.464 = 0.466
```

A κ of `0.47` falls in the "moderate agreement" band of the Landis-Koch convention. The two filters agree substantially more than chance would predict but also disagree meaningfully on roughly a quarter of the corpus.

**Direction of disagreement.** The two off-diagonal cells of the contingency table characterise the disagreement asymmetry:

- **Regex-rejected, LLM-accepted** (1,491 articles, 7.6% of total). These are articles the regex flagged as having an invalid body (`body_valid=0`) but the LLM judged usable for sentiment extraction (`usable=1`). They represent the regex's false negatives: substantive content that the keyword-and-length heuristic discarded.

- **Regex-accepted, LLM-rejected** (3,366 articles, 17.2% of total). These are articles the regex accepted as having valid bodies (`body_valid=1`) but the LLM judged unusable (`usable=0`). They represent the regex's false positives: content that satisfied the keyword and length criteria but is not substantively about WTI markets.

The asymmetry is informative. The regex tends to be permissive on long-form-but-non-substantive content, because it cannot semantically judge whether an energy-keyword-containing article is about oil markets or about an unrelated topic, but slightly restrictive on short-form-but-substantive content, because the 400-character length threshold is a blunt criterion. The LLM judgment, having semantic access to the article body, behaves in the opposite way on both axes: it can recognise short substantive content and reject long non-substantive content.

**Manual audit of disagreement cases.** To characterise the two disagreement categories beyond their counts, a sample of articles from each off-diagonal cell was manually inspected and hand-labelled for true usability.

The regex-accepted/LLM-rejected cases (the regex's false positives) are dominated by keyword-collision off-topic content. Several distinct sub-patterns recur: agricultural commodities sharing the "oil" token (a tariff-free palm oil shipment to the United Kingdom, a Canadian canola export agreement); consumer-facing explainers that mention oil products without carrying market signal (an article comparing heating-oil and gasoline retail prices); companies whose names contain energy terms but whose business is unrelated to oil markets (a semiconductor-equipment manufacturer trading up on analyst upgrades); commodity-adjacent market reports outside the oil complex (a global carbon and graphite market study); and generic equity-market wrap articles or stock-pick listicles that mention oil only in passing. In these cases the LLM's semantic rejection is correct and the regex's keyword match is a false positive. The manual labels confirmed the LLM's `usable=0` judgment for the large majority of inspected cases in this cell.

The regex-rejected/LLM-accepted cases (the regex's false negatives) are dominated by short but substantive WTI news that falls below the regex's 400-character body threshold. A representative example is a 320-character Bloomberg brief reporting that OPEC+ held its existing production plan despite political pressure to lower prices — squarely on-topic for WTI but too short for the regex to accept. Another is a brief macro note linking falling oil prices to reduced recession odds. In these cases the LLM's acceptance is correct and the regex's length-based rejection is a false negative.

The manual audit also surfaced a residual error mode in the LLM filter itself. A small number of articles marked `usable=true` by the LLM are, on inspection, not genuinely WTI-relevant. The clearest example is an article on Canada rolling back tariffs on Chinese electric vehicles, which the LLM accepted and assigned a non-trivial sentiment score and a `[demand, macro, other]` event-type tag, despite the article being about automotive trade policy with no material implication for crude oil markets. Scraping artefacts also occasionally slipped through: at least one accepted "article" consisted not of body text but of a list of unrelated headlines captured during scraping. These cases indicate that the LLM filter, while substantially more accurate than the regex on the regex's two failure modes, is not itself a ground-truth oracle.

The audit therefore supports a measured conclusion rather than a claim of LLM superiority in all respects. Neither filter is perfect. The LLM filter is preferred because its residual error rate is lower and its errors are less systematic than the regex's keyword-collision and length-threshold failures: the regex fails predictably on entire categories of content (any article containing an energy keyword regardless of topic, any substantive article below the length threshold), whereas the LLM's failures are sparser and less structured. The existence of LLM false positives, however, is a genuine limitation that bears on the interpretation of all Phase 2 results and is revisited in the discussion (Chapter 5).

**Methodological consequences.** The choice of filter affects which articles enter downstream modeling, and the asymmetry above is not negligible. On the Phase 2 dataset:

1. Under the regex filter, 13,550 articles would be retained for modeling. Of these, the LLM judges that 3,366 (24.8%) are not substantively usable.
2. Under the LLM filter, 11,675 articles are retained. Of the 7,944 articles excluded by the LLM, 3,366 were regex-accepted (the regex's false positives) and 4,578 were also regex-rejected (joint rejections).
3. The LLM filter additionally accepts 1,491 articles the regex would have rejected.

The two filters are not nested. Using the LLM filter is not equivalent to applying the regex filter and then refining it; the LLM filter accepts a set of articles that overlaps but is not a subset of the regex's acceptance set. For the Phase 2 analyses, the LLM filter is the canonical downstream filter, applied uniformly across all subsections that use the Phase 2 corpus.

**Implication for cross-phase comparison.** Direct comparison of Phase 1 and Phase 2 results is complicated by the filter migration: Phase 1 analyses operate on the 13,690-article modeling dataset (regex-filtered with title-only fallback for articles failing the filter), while Phase 2 analyses operate on the 11,675-article modeling-ready subset (LLM-filtered, no title-only fallback). Section 4.3.8 discusses how this dataset shift interacts with the model comparison and whether any observed change between TFT v1 and v2 can be attributed cleanly to the filter migration as opposed to other Phase 2 changes.

### 4.3.4 Inter-model calibration of LLM features

The LLM-extracted features described in Section 4.3.2 are the substrate of all Phase 2 modeling, so their quality must be validated before they are relied upon. Section 3.7 set out the methodology for this validation: an inter-model calibration in which the same articles, with the same extraction prompt and tool schema, are scored by Claude Haiku and by a GPT-family reference model, with disagreement between two models from different developer families treated as a signal of either genuine ambiguity or systematic bias. This subsection reports the first calibration result, obtained on the initial Haiku v1 schema, and the diagnostic that it prompted.

**Setup.** The calibration was run on a stratified sample of 30 articles drawn from the three body-validity strata described in Section 3.7.2. Each article was scored by Claude Haiku 4.5 via the production extraction code, and by GPT-5 model accessed through the free ChatGPT web interface using the identical system prompt and field definitions. The reference model is not claimed to be a state-of-the-art configuration; the purpose of the comparison is cross-family agreement, not benchmarking a specific model. The two output sets were aligned by article and compared field by field, with agreement metrics computed symmetrically as described in Section 3.7.4.

**Result on the v1 schema.** The Haiku v1 schema produced a single composite `sentiment_score` as its primary directional feature, alongside `magnitude`, `event_type`, `entities`, `certainty`, `price_direction`, and `time_horizon`. The calibration revealed acceptable agreement on the usability judgment but poor agreement on the composite sentiment score:

- **Usability agreement**: the two models agreed on the `usable` flag for 26 of 30 articles (87%). The four disagreements were borderline corporate or analyst-commentary articles that one model accepted and the other rejected — the kind of marginal content the stratified sampling was designed to surface.
- **Sentiment correlation**: across the usable articles, the Pearson correlation between the two models' `sentiment_score` values was only **0.39**. For a feature intended to carry the primary directional signal into downstream models, this is a weak agreement.
- **Sign disagreements**: the two models assigned opposite-signed sentiment scores on **4 of 13 usable articles (31%)**. A sign disagreement is the most consequential form of disagreement, since it means the two models read the same article as carrying opposite directional implications for WTI price.

**Diagnostic.** The sign disagreements were not randomly distributed across the sample. They concentrated on high-magnitude geopolitical events — articles about Iran-US tensions, OPEC supply politics, and similar cases where the relationship between the news event and the oil price is economically non-trivial. Inspecting these cases identified a consistent root cause: the composite `sentiment_score` conflates two distinct judgments that an annotator (human or model) must make about an oil-market news article.

The first judgment is the **valence of the event itself**: is this good news or bad news for the situation being described? The second is the **directional implication for WTI price**: does this event push crude prices up or down? For most articles these two judgments align, and a single sentiment score captures both adequately. But for high-magnitude geopolitical events they can diverge sharply. Consider an article reporting an escalation that threatens oil supply: the event is unambiguously negative in valence (conflict, disruption, threat), yet its directional implication for WTI price is _positive_ (supply threat drives prices up). A model weighing valence will score such an article negative; a model weighing price impact will score it positive. The composite `sentiment_score` does not tell the model which judgment to make, so the two reference models resolve the ambiguity differently, producing the observed sign disagreements precisely on the cases where the distinction matters most.

This diagnostic is the pivot point of the Phase 2 methodology. A sentiment score that is unreliable specifically on high-magnitude geopolitical events is unreliable on exactly the articles that matter most for a WTI liquidity model, since those events are the ones that move the market. A correlation of 0.39 on the primary directional feature is not adequate for downstream modeling, and the failure is structural rather than incidental: it stems from asking a single number to encode two separable economic judgments.

**Response.** The diagnostic motivated a schema revision rather than a prompt tweak. If the problem is that one number is being asked to encode two judgments, the solution is to decompose the judgment into its separable components. Section 4.3.6 documents the resulting channel decomposition — the introduction of `supply_impact`, `demand_impact`, and `risk_premium` as orthogonal economic channels — and the re-calibration that validates it. Before that, however, the first deep-learning model of the project, TFT v1, was trained on the v1 schema; its results and limitations are the subject of the next subsection.

### 4.3.5 Temporal Fusion Transformer v1

The first Temporal Fusion Transformer (TFT v1) is the project's first deep-learning model and the first to exploit the richer Haiku-extracted feature set. It was trained on the Phase 1 corpus, processed through the Haiku v1 schema, with the macro covariates DXY and VIX added. This subsection reports its training outcome, its interpretation outputs, and the limitations that motivate the subsequent refinements of Phase 2.

**Training setup.** The model follows the architecture described in Section 3.6: a 48-hour encoder window, a one-hour prediction horizon, `hidden_size=32`, `attention_head_size=4`, `dropout=0.1`, and a quantile loss, totalling approximately 113,000 trainable parameters. Training was performed on a Google Colab T4 GPU via the `pytorch-forecasting` library. The dataset comprised 10,797 hourly rows with an 80/20 temporal split (approximately 8,637 training hours and 2,160 validation hours). Early stopping on validation loss triggered at epoch 21, with a best validation loss of **0.204**.

The input feature set combined three groups. The LLM features (from the Haiku v1 schema) were `sentiment_score`, `magnitude`, `event_type` (integer-encoded as `event_type_num`), `certainty`, `price_direction` (as `price_direction_num`), `time_horizon` (as `time_horizon_num`), and `n_articles`.

The market-context features were `log_volume`, `price_range`, `log_return`, `amihud`, `dxy`, and `vix`.

The temporal covariates were `hour`, `day_of_week`, `month`, `is_wednesday`, and `is_us_session`.

**Feature importance.** The Variable Selection Network assigns a learned importance weight to each input feature, averaged over the validation set. The result is dominated by a single feature:

| Feature              |  Importance | Group  |
| -------------------- | ----------: | ------ |
| `sentiment_score`    |        0.53 | News   |
| `log_volume`         |       0.095 | Market |
| `dxy`                |       0.075 | Market |
| `log_return`         |       0.055 | Market |
| `event_type_num`     |       0.030 | News   |
| `magnitude`          |       0.025 | News   |
| (remaining features) | < 0.02 each | —      |
| `vix`                |       0.005 | Market |

![IRF Sentiment to Volume](../../../04_outputs/figures/tft_feature_importance.png)

**Figure 4.5: TFT v1 feature importance, news features (red) versus market features (blue).** The `sentiment_score` feature dominates with an importance weight of 0.53, more than five times the next-ranked feature. The market-context features (`log_volume`, `dxy`, `log_return`) form a second tier, and the remaining news and temporal features carry minor weight.

Two observations stand out. First, **`sentiment_score` is the single most important feature by a large margin**, carrying an importance weight more than five times that of the next feature. This is a strong validation of the LLM extraction approach: the continuous LLM-derived sentiment carries far more signal for the model than any single market-context variable, and dramatically more than the FinBERT three-class label could have provided. Second, **`vix` is effectively negligible** at an importance of 0.005, while `dxy` ranks third overall. The interpretation is that for WTI specifically, the dollar index already captures most of the macro-risk signal that VIX would contribute, so VIX adds little once DXY is present. This is a useful finding for the feature design of TFT v2.

A further observation concerns the integer-encoded categorical features. `event_type_num` ranks modestly (0.030), above `magnitude`, suggesting that the _category_ of a news event is more informative for volume than its _size_. However, `event_type_num`, `price_direction_num`, and `time_horizon_num` were all encoded as continuous integers rather than as proper categoricals in TFT v1. This means the model was forced to treat, for example, `event_type=3` as numerically "larger" than `event_type=1`, an ordering that has no economic meaning. Their low importance weights may therefore understate their true informativeness; the integer encoding is a limitation addressed in TFT v2 (Section 4.3.7).

**Attention and the lag structure (RQ1).** The interpretable multi-head attention layer reveals which historical hours the model attends to when predicting volume. Averaged over the validation set, the attention weights peak at **lag −4h** (weight 0.0296), with the five most-attended lags being −4h (0.0296), −2h (0.0269), −28h (0.0262), −5h (0.0259), and −27h (0.0257).

![TFTv1 attention weights](../../../04_outputs/figures/tft_attention_weights.png)

**Figure 4.6: TFT v1 encoder attention by lag.** Mean attention weight over the 48-hour encoder window, with the peak at lag −4h marked. Two distinct attention clusters are visible: a short-term cluster between −2h and −5h, and a secondary cluster around −27h to −28h.

The attention pattern carries two findings relevant to RQ1.

The first is a **short-term news-absorption window** between lag −2h and −5h, peaking at −4h. This is independent corroboration of the lag OLS finding from Phase 1 (Section 4.2.4), which identified a peak news impact at lag +6h. The two methods are very different — the lag OLS is a sequence of single-lag linear regressions on discrete sentiment events, while the TFT learns a single nonlinear function over the full input history and exposes its temporal focus through attention — yet both identify the same four-to-six-hour window as the most informative horizon for news-driven volume. The slight difference in the precise peak (TFT −4h versus OLS +6h) is within the resolution of the two methods and does not detract from the convergence: both place the dominant news-impact horizon several hours after publication rather than contemporaneously.

The second is a **secondary attention cluster around lag −27h to −28h**, which we term a "daily memory" effect. The model attends not only to the recent past but also to conditions roughly 24 to 28 hours earlier — the same time of day on the previous trading day. This is consistent with daily market cycles: the volume at a given hour is partly predicted by the volume at the same hour on the previous day. This effect was invisible to the Phase 1 lag OLS, which examined a maximum lag of 12 hours, and is an example of the TFT extracting structure that the narrower OLS window could not reach. The interpretation of the daily memory effect — whether it reflects traders revisiting positions at the same time each day, overnight cycle effects, or an interaction between the encoder window and the `is_us_session` indicator — is taken up in the discussion (Chapter 5).

**Directional asymmetry (RQ2).** To test the bearish-versus-bullish asymmetry directly, the validation samples were split by dominant sentiment direction and the model's mean predicted `log_volume` compared across the three categories:

| Sentiment direction | Mean predicted `log_volume` |
| ------------------- | --------------------------: |
| Bearish             |                       8.775 |
| Bullish             |                       8.700 |
| Neutral             |                       8.875 |

Two features of this result are notable. First, **bearish predicted volume exceeds bullish predicted volume** (8.775 versus 8.700), consistent in direction with the lag OLS asymmetry finding of Section 4.2.4 and with the contemporaneous OLS of Section 4.2.3. The TFT, trained as a nonlinear predictor with no asymmetry constraint imposed, independently reproduces the bearish-greater-than-bullish ordering. Second, a two-sample t-test for the difference between bearish and bullish predicted volume returns **p = 0.56**, which is not statistically significant. The validation split provides only a few hundred hours per sentiment category, leaving the directional test underpowered. The asymmetry is therefore directionally consistent across all of the project's models but reaches statistical significance only in the lag OLS (Section 4.2.4), which operates on the full 13,690-article sample. The lag OLS remains the canonical evidence for RQ2; the TFT corroborates its direction but cannot independently establish its significance.

A third observation is counterintuitive and worth noting: **neutral-sentiment hours show the highest predicted volume of the three categories** (8.875). One interpretation is that genuine market uncertainty — hours where the news flow is present but directionally ambiguous — drives more trading activity than clearly directional news. This is speculative and not central to the research questions, but it is a candidate observation for the discussion.

**Limitations of TFT v1.** The TFT v1 results validate the core modelling approach: the LLM-derived sentiment dominates feature importance, the attention pattern corroborates the lag OLS, and the directional asymmetry is reproduced. But the model has five limitations that collectively motivate the Phase 2 refinements that follow:

1. **The directional asymmetry test is underpowered.** With only a few hundred validation hours per sentiment category, the t-test cannot establish significance (p=0.56). A larger dataset is needed to test whether the TFT can independently confirm the RQ2 asymmetry.
2. **Categorical features are integer-encoded.** `event_type_num`, `price_direction_num`, and `time_horizon_num` were passed as continuous integers, imposing a meaningless numerical ordering on categorical values and likely understating their informativeness.
3. **The model has no entity-level awareness.** The Haiku v1 schema extracted entities, but TFT v1 did not use them. A geopolitical event involving Iran and a routine inventory report are treated identically by the model except through their continuous feature values.
4. **The training data is Phase 1 only.** TFT v1 was trained on the pre-war corpus ending February 2026, missing the post-war period with its elevated news intensity and price volatility.
5. **The sentiment feature is the unreliable composite.** TFT v1 used the Haiku v1 `sentiment_score`, which the calibration of Section 4.3.4 showed to be unreliable on high-magnitude geopolitical events (correlation 0.39, 31% sign disagreement). The single most important feature in the model is the one the calibration flagged as least trustworthy.

The fifth limitation is the most consequential, and it connects directly to the calibration finding. The TFT v1 leans overwhelmingly on `sentiment_score` (53% importance), yet that score is exactly the feature the inter-model calibration identified as conflating event valence with price direction. The natural response is to give the model a more reliable directional signal — the channel decomposition introduced in the next subsection — and then retrain. Section 4.3.6 documents the channel decomposition and its validation; Section 4.3.7 documents the resulting TFT v2.

### 4.3.6 Channel decomposition response

The inter-model calibration of Section 4.3.4 identified a structural weakness in the Haiku v1 schema: the composite `sentiment_score` conflated event valence with directional price impact, producing a Pearson correlation of only 0.39 between Haiku and the GPT reference and a 31% sign-disagreement rate on the cases that matter most. The TFT v1 of Section 4.3.5 was trained on this unreliable score and assigned it 53% feature importance, meaning the model's strongest input was its least trustworthy feature. The natural response is to give both the model and downstream analyses a more reliable directional signal. This subsection documents the schema revision that produced that signal — the channel decomposition — and the re-calibration that validates it.

**Diagnosis and design principle.** The conflation diagnosed in Section 4.3.4 has a structural cause: an oil-market news article carries up to three economically distinct judgments that a sentiment annotator must make. Does the event imply more or less _supply_? Does it imply more or less _demand_? Does it shift _geopolitical or operational risk_? A composite sentiment score asks a single number to encode the resolution of all three. When the three judgments align (a routine inventory build is bearish-supply, neutral-demand, low-risk, hence bearish overall), one number suffices. When they diverge — a supply shock that is bearish for the event itself but bullish for prices, a geopolitical de-escalation that is positive in valence but bearish-supply via sanctions relief — the composite breaks down.

The design principle is therefore to decompose the single sentiment judgment into its separable economic components, ensuring that each new score captures one factual claim rather than a blend. The schema is engineered so that each channel has a clear interpretive anchor in the prompt and a single direction along which it varies.

**Schema changes.** The Haiku v2 schema modifies the v1 schema in five ways. Three are additive, one is subtractive, and one restructures an existing field.

1. **Three orthogonal economic channels are added**, each a continuous score on `[-1, +1]`:
   - `supply_impact`: positive when the event implies more oil available to market (sanctions lifted, OPEC adding barrels, infrastructure online); negative when the event implies less (refinery outages, OPEC cuts, infrastructure attacks).
   - `demand_impact`: positive when the event implies stronger oil consumption (Fed easing, China stimulus, growth surprise); negative when it implies weaker consumption (recession signals, Fed hawkishness, lockdowns).
   - `risk_premium`: positive when the event implies elevated geopolitical or operational risk priced into oil (military strikes, sanctions, escalation); negative when it implies de-escalation.

2. **A `usable` boolean is promoted to a required schema field** rather than implicit in the prompt. This supports the Phase 2 filter migration documented in Section 3.4.2 and Section 4.3.3.

3. **`event_type` is changed from a single categorical label to an array of one to three labels**, ordered by salience within the article. The change allows the schema to represent articles that are jointly geopolitical and supply-related (the typical case for Iran sanctions, for example) without forcing the model to commit to a single dominant category.

4. **`price_direction` is dropped from the schema**. Inspection of the v1 extraction batch showed that `price_direction` and the sign of `sentiment_score` agreed on all but a handful of articles (approximately 16 cases out of 12,024 differed meaningfully). The field added no information not already captured by `sentiment_score` and contributed no useful signal in TFT v1 (importance approximately 0.008). The decomposition into three channels also provides a richer directional substrate, making `price_direction` redundant.

5. **The composite `sentiment_score` is retained** rather than replaced by the channels. Three reasons support this. First, retention preserves continuity with the Phase 1 headline bias experiment (Section 4.2.2), which depends on a single sentiment value. Second, the composite enables a direct ablation study comparing single-feature performance against the decomposed channels in TFT v2 (Section 4.3.7). Third, it serves as a hedge against the channels turning out to be individually noisy at production scale.

**Prompt revision and re-extraction of the calibration sample.** The system prompt and tool schema documented in Appendix A were revised to incorporate the changes above, with explicit anchors in the prompt for each new channel's interpretation. The revised schema was then re-applied to the same 30-article stratified sample used in the v1 calibration of Section 4.3.4. Using the same sample is methodologically important: it isolates the effect of the schema change from the effect of sample variation. Both Haiku and GPT-5 were re-scored on the 30 articles with the new prompt and schema.

**Re-calibration result.** The re-calibration shows substantial improvement on every metric reported for the v1 schema, with the largest improvement on the field that was most problematic. Table 4.5 reports the v1 and v2 figures side by side.

| Metric                               | v1 schema         | v2 schema            |
| ------------------------------------ | ----------------- | -------------------- |
| `usable` agreement                   | 26/30 (87%)       | 27/30 (90%)          |
| `sentiment_score` correlation        | 0.39              | **0.88**             |
| `sentiment_score` sign disagreements | 4/13 usable (31%) | **1/14 usable (7%)** |
| `supply_impact` correlation          | —                 | 0.94                 |
| `supply_impact` MAD                  | —                 | 0.11                 |
| `supply_impact` sign disagreements   | —                 | 0                    |
| `demand_impact` correlation          | —                 | 0.96                 |
| `demand_impact` MAD                  | —                 | 0.05                 |
| `demand_impact` sign disagreements   | —                 | 0                    |
| `risk_premium` correlation           | —                 | 0.82                 |
| `risk_premium` MAD                   | —                 | 0.13                 |
| `risk_premium` sign disagreements    | —                 | 1                    |

The headline numbers are:

- **`sentiment_score` correlation improves from 0.39 to 0.88**, a 49-percentage-point gain. The composite score, when supplied alongside the three orthogonal channels in the prompt, becomes far more consistent across models. The decomposition does not replace the composite — it provides interpretive context that disciplines how the composite is computed.
- **`sentiment_score` sign disagreements drop from 4/13 to 1/14.** The single remaining disagreement on the sentiment composite is on a borderline case rather than the high-magnitude geopolitical events that dominated the v1 disagreements.
- **Each of the three new channels achieves correlation above 0.82**, with `supply_impact` and `demand_impact` reaching 0.94 and 0.96 respectively. The mean absolute differences are all under 0.13. Sign disagreements are zero for the two best-correlated channels and one for `risk_premium`.

The improvement on the composite `sentiment_score` is, in some respects, the more interesting result. The channels are new features and one would expect them to be designed for high agreement. But the composite was not modified — it is still a single signed number on [-1, +1]. The improvement on the composite reflects the prompt-engineering effect of asking the model to first decompose the article into its separable components and only then to produce a composite. The decomposed reasoning structure improves agreement on every output it touches, not only on the new channels themselves.

**Orthogonality of the channels.** A second methodological concern is whether the three channels are genuinely orthogonal or whether they collapse into the same underlying signal under different names. The within-model pairwise correlations of the channel scores, computed across the 30 calibration articles, are all below 0.5 in absolute value for both Haiku and the GPT reference. The channels capture genuinely different economic dimensions of the same news content rather than rephrasing a single judgment, supporting their use as independent features in downstream modeling.

**A residual weakness in the Haiku v2 outputs.** The re-calibration also surfaced a behavioural difference between the two models that did not appear in the headline metrics. On three of the 14 usable articles, Haiku assigned all three channel scores to exactly zero while the GPT reference assigned non-trivial channel values to the same articles. The cases were articles with genuine but modest WTI relevance — for example, articles touching on macro forces with no direct supply or demand implication, or articles describing low-intensity geopolitical activity. Haiku appeared to punt on these articles by zeroing the channels rather than committing to a small-magnitude judgment, while GPT was willing to assign small-magnitude values. The corresponding `sentiment_score` and `magnitude` fields were also subdued on these cases, so the articles do not silently dominate downstream features, but the pattern indicates a slight conservatism in Haiku's channel assignments that is worth noting.

**Implication for downstream modeling.** With the channels validated, the production extraction batch documented in Section 4.3.2 was run on the full Phase 2 corpus of 19,619 articles using the Haiku v2 schema. The resulting feature set is the substrate for TFT v2 (Section 4.3.7), which incorporates the three channels alongside the macro covariates, proper categorical encodings for `event_type` and `time_horizon`, and entity-level binary flags. The Phase 1 vs Phase 2 comparison is then taken up in Section 4.3.8.
