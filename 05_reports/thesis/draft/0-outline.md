# Thesis Outline

**Title:** News-Driven Liquidity Dynamics in WTI Crude Oil Futures: A Channel-Decomposition Approach

**Author:** Enrique Favila\
**Supervisor:** Dr. Lejla Batina\
**Institution:** Radboud University\
**Host:** Hammer Market Intelligence

## Purpose of this document

Structural reference for the thesis. The work investigates how news propagates into WTI crude oil futures liquidity at hourly resolution, using a channel decomposition of LLM-extracted news features (supply, demand, risk premium) to characterize the lag structure (RQ1) and the bearish/bullish asymmetry (RQ2) of the liquidity response.

## Chapter 1 - Introduction

**Purpose:** What is the problem, current research questions, contributions.

- 1.1 Motivation
  - News shapes commodity markets, but most empirical work operates at daily resolution, missing the intra-day dynamics where information actually propagates
  - WTI crude oil specifically has a rich event structure (geopolitics, OPEC, EIA inventories, macro) that makes it well-suited for hourly-resolution analysis
- 1.2 Problem statement
  - How does news propagate into WTI liquidity at hourly resolution?
  - What is the lag structure, the directional asymmetry, the role of macro and event-type features?
- 1.3 Research questions
  - **RQ1**: lag structure of news impact on liquidity
  - **RQ2**: bearish vs bullish asymmetry in liquidity response
- 1.4 Contributions
  - (a) Empirical lag-and-asymmetry finding for WTI at hourly resolution, with peak news impact at lag +6h and a consistent bearish > bullish asymmetry. Supported by lag OLS results (4.2.4) and confirmed by TFT v2 attention patterns (4.3.7).
  - (b) Methodological contribution: a decomposition of LLM-extracted news features into orthogonal economic channels (supply, demand, risk premium), validated by inter-model calibration. The decomposition improves cross-model agreement on news sentiment by roughly 50 percentage points relative to a single composite score. Supported by 4.3.2–4.3.5.
  - (c) Comparative finding on news content representation: title-only and full-body sentiment extraction disagree on 41.6% of articles, with systematic asymmetry. Supported by 4.2.2.
- 1.5 Structure of the thesis

## Chapter 2 - Background

**Purpose:** Describe everything needed to follow Chapters 3 and 4.

- 2.1 Terminology
  - 2.1.1 WTI crude oil futures and the EIA inventory cycle
  - 2.1.2 Liquidity measures: log volume, Amihud illiquidity, Parkinson range
  - 2.1.3 NLP for finance: sentiment classification, structured extraction, prompt engineering
  - 2.1.4 Temporal Fusion Transformer: variable selection network, multi-head attention
- 2.2 State of the Art
  - 2.2.1 News impact studies on commodity markets (mostly daily-resolution)
  - 2.2.2 Financial NLP: FinBERT and its limitations
  - 2.2.3 LLM-based feature extraction in finance
  - 2.2.4 Temporal Fusion Transformer applications in financial time series

## Chapter 3 - Methods

**Purpose:** Describe the techniques and methodology used in the project. Each phase in Chapter 4 invokes these techniques.

- 3.1 Data sources and acquisition
  - Commodity selection (with the WTI vs sugar screening note)
  - Market data via yfinance (CL=F, DXY, VIX)
  - EIA Weekly Petroleum Status Reports (WCRSTUS1)
  - GDELT headline scraping (eight Phase 1 queries, five Phase 2 additions)
  - Body retrieval via BeautifulSoup
  - Persistence to SQLite
- 3.2 Temporal alignment
  - 3.2.1 Ceiling assignment to the next trading hour
  - 3.2.2 Forward-assignment of off-hours articles
  - 3.2.3 Joined output
- 3.3 News feature extraction approaches
  - 3.3.1 FinBERT sentiment extraction (Phase 1)
  - 3.3.2 Structured LLM extraction with Claude Haiku (Phase 2 covers both Haiku v1 and Haiku v2 schemas)
  - 3.3.3 Tool-use API and schema enforcement
  - 3.3.4 Channel decomposition (supply, demand, risk premium)
  - 3.3.5 Magnitude, event type, entities, certainty, time horizon
- 3.4 Filtering strategies
  - 3.4.1 Regex/keyword heuristic (`body_valid`, Phase 1)
  - 3.4.2 LLM-judged `usable` flag (Phase 2)
  - 3.4.3 Comparative analysis
- 3.5 Statistical models
  - 3.5.1 Lag OLS specification
  - 3.5.2 Vector autoregression specification
- 3.6 Temporal Fusion Transformer
  - 3.6.1 Architecture overview
  - 3.6.2 Input typing
  - 3.6.3 Forecast configuration
  - 3.6.4 Architectural hyperparameters
  - 3.6.5 Training procedure
  - 3.6.6 Interpretation outputs
- 3.7 Inter-model calibration methodology
  - 3.7.1 The validation problem and our approach
  - 3.7.2 Sample design
  - 3.7.3 Extraction procedure
  - 3.7.4 Agreement metrics
  - 3.7.5 Use of the calibration outcome

## Chapter 4 - Experiments

**Purpose:** Document the work done, what was found, and how each finding motivated the next step. There exists so far 2 phases.

### 4.1 Experimental setup

- Hardware
- Software stack
- Data partition strategy: temporal holdout (last 20% as validation)
- Reproducibility: seeds, version-locked dependencies, deterministic prompts

### 4.2 Phase 1 - Initial pipeline and baseline modeling

**Phase scope:** WTI market data, GDELT scrape across the Phase 1 query set (eight queries, March 2024 to February 2026), FinBERT sentiment extraction, regex-based `body_valid` filter. The phase covers the headline bias experiment, the contemporaneous and lag OLS regressions, and the VAR specification. The phase establishes the empirical findings on RQ1 and RQ2 and exposes the limitations that motivate the Phase 2 NLP redesign and the move to deep-learning modeling.

- 4.2.1 Phase 1 data and feature setup
  - Coverage period, GDELT scrape (51,948 → 16,326 → 7,756 valid bodies + 5,934 title-only fallback = 13,690 modeling-ready)
  - FinBERT extraction, sentiment label mapping
  - EIA inventory features (is_eia_release, eia_surprise)
  - body_valid regex filter (described in detail in §3.4.1)
- 4.2.2 Headline bias experiment
  - Methodology: FinBERT title-only vs title-plus-body on the 7,756 body_valid=1 articles
  - Result: 41.6% sentiment disagreement, χ²=2050, p<0.001
  - Asymmetry: titles overrepresent negative sentiment
  - **Figure: headline_bias_comparison.png**
  - Implication: full bodies preferred for downstream feature extraction
- 4.2.3 Contemporaneous OLS
  - Methodology: news sentiment at hour t regressed on log_volume at hour t
  - Result: R² < 0.001, neither bearish nor bullish significant
  - Interpretation: news does not move volume instantaneously at the moment of publication
  - This null result motivates the lag analysis
- 4.2.4 Lag OLS, first answer to RQ1 and RQ2
  - Methodology: log_volume at t+k regressed on bearish/bullish indicators at t, for k ∈ {1, 2, 3, 4, 6, 8, 12}
  - Result: peak impact at lag +6h (β ≈ 0.25, p < 0.001), consistent bearish > bullish asymmetry
  - **Figure: lag_coefficients.png**
  - Canonical evidence for RQ1 and RQ2
- 4.2.5 Vector Autoregression
  - Methodology: VAR on hourly time series of sentiment and liquidity
  - Result: impulse response function suggests modest persistent effect but with wide error bands
  - **Figure: irf_sentiment_to_volume.png**
  - Abandonment rationale: ~50% zero-sentiment hours (signal sparsity), VAR estimator underidentified, IRF confidence intervals span zero at most horizons
  - This motivates a model that handles sparse event-driven signal natively → TFT (Phase 2)
- 4.2.6 Phase 1 summary
  - Findings: lag +6h peak, bearish > bullish asymmetry, headline bias
  - Limitations driving Phase 2:
    - FinBERT cannot represent event type, entities, or causal direction
    - Regex filter has documented false positives and false negatives
    - VAR cannot identify dynamics on sparse sentiment signal
    - Phase 1 dataset confined to pre-war coverage window
    - Models lack macro covariates (DXY, VIX)

### 4.3 Phase 2 - Refined pipeline and improved modeling

**Phase scope:** Migration from FinBERT to Claude Haiku for richer structured feature extraction, addition of DXY and VIX as macro covariates, and the introduction of the Temporal Fusion Transformer as the primary deep-learning model. The phase opens with the Haiku v1 schema and the first TFT training (TFT v1), then iterates: inter-model calibration exposes a weakness in the composite sentiment score, the schema is decomposed into orthogonal economic channels (Haiku v2), the news corpus is extended through May 2026, and TFT v2 is trained on the refined feature set. The phase ends with a direct comparison of TFT v1 and v2 and a robustness check on the filter migration.

- 4.3.1 Phase 2 data and feature setup
  - Migration rationale (FinBERT limitations from Phase 1)
  - Addition of DXY and VIX as macro covariates
  - Coverage extension to post-war period (through May 2026)
  - Initial corpus: same articles as Phase 1, processed through Haiku v1 for the first TFT
- 4.3.2 LLM feature extraction with Claude Haiku
  - Tool-use API with strict JSON schema (no parsing brittleness)
  - Initial schema (Haiku v1): sentiment_score, magnitude, event_type, entities, certainty, price_direction, time_horizon
  - Prompt design and iteration
  - Cost and operational details (Batches API, prompt caching)
- 4.3.3 LLM usable flag and filter comparison
  - Replacing the regex heuristic with an LLM judgment of content usability
  - Comparison: body_valid (13,550 accepted) vs usable (11,675 accepted), Cohen's κ
  - False-positive cases (regex rejected, LLM accepted)
  - False-negative cases (regex accepted, LLM rejected)
  - Methodological contribution: LLM-based filtering for noisy news corpora
- 4.3.4 Inter-model calibration of LLM features
  - Methodology: 30-article stratified sample, scored by Haiku and by a GPT-family reference model with identical prompt
  - First calibration result: sentiment_score correlation 0.39 between models, sign disagreement on 4/13 usable articles (31%)
  - Diagnostic: disagreements concentrate on high-magnitude geopolitical events
  - Root cause analysis: sentiment_score conflates event valence with directional price impact
- 4.3.5 Temporal Fusion Transformer v1
  - First TFT training, on the Phase 1 corpus with Haiku v1 features and macro covariates
  - Architecture and training setup (Section 3.6)
  - Results: val_loss 0.204, attention peak at −4h (consistent with lag OLS +6h), sentiment_score at 53% feature importance
  - "Daily memory" effect at -27/-28h in attention
  - Limitations:
    - Directional asymmetry test underpowered (p=0.56)
    - event_type, price_direction, time_horizon encoded as continuous integers (no real signal)
    - No entity-level awareness
    - Pre-war data only
    - Inter-model calibration finding (§4.3.4) had not yet motivated the channel decomposition
- 4.3.6 Channel decomposition response (Haiku v2 schema)
  - Schema extension: supply_impact, demand_impact, risk_premium, each on [-1, +1]
  - Each channel is a single factual judgment, designed to be orthogonal
  - sentiment_score retained for continuity with the headline bias experiment and as a composite
  - price_direction dropped (empirically redundant with sentiment_score)
  - event_type changed from single string to array of 1–3 categories
  - usable flag added for filtering (Phase 2 §3.4.2)
  - Second calibration result:
    - supply_impact: r=0.94, MAD 0.11
    - demand_impact: r=0.96, MAD 0.05
    - risk_premium: r=0.82, MAD 0.13
    - sentiment_score (as side effect): r=0.39 → r=0.88
    - Sign disagreements: 4/13 → 1/14
  - Within-model orthogonality preserved (all pairwise |r| < 0.5)
- 4.3.7 Full LLM extraction batch and TFT v2
  - Execution: 19,619 articles via Batches API on the expanded corpus
  - Outcome: 11,675 usable (59.5%), channel statistics
  - TFT v2 architecture changes from v1:
    - Proper categorical encodings (event_type, time_horizon)
    - Multi-hot entity binary flags (top 15 entities)
    - Three orthogonal channel features
    - Trained on the expanded 22,795-aligned dataset
  - Results: val_loss, attention patterns, feature importance per channel, still pending
- 4.3.8 Phase 1 vs Phase 2 (v1 vs v2) comparison
  - Table: val_loss, top-3 features by importance, asymmetry test p-value
  - Discussion of which Phase 2 changes contributed most to which improvements, if any

### 4.4 Robustness checks

- 4.4.1 Filter comparison (body_valid vs usable, with the κ analysis from 4.3.3)
- 4.4.2 Optional: alignment robustness (lag analysis re-run with floor alignment instead of ceiling, to verify the lag +6h finding is not an artifact of the alignment rule)

## Chapter 5 - Discussion

**Purpose:** Interpret the findings, contextualize them in the literature, identify limitations.

- 5.1 Findings on RQ1 (lag structure)
  - The lag +6h peak in OLS, confirmed by TFT v2 attention
  - What this means about how news information propagates
- 5.2 Findings on RQ2 (asymmetry)
  - Bearish > bullish in OLS is statistically robust
  - TFT directional asymmetry test underpowered; lag OLS retains canonical evidence
- 5.3 Methodological contributions
  - Channel decomposition as a response to inter-model calibration failure
  - LLM-as-filter as an alternative to regex heuristics for noisy news corpora
  - Phase-based pipeline progression as a reproducible template for similar empirical work
- 5.4 Limitations
  - Inter-LLM calibration is methodologically weaker than human ground truth
  - GDELT corpus skews retail-aimed news rather than institutional-grade feeds
  - 24-month dataset with regime change (Iran war) introduces non-stationarity
  - Single-commodity scope; results may not generalize to other commodities or asset classes
- 5.5 Future work
  - Cross-commodity spillovers (out of scope for this thesis)
  - Larger calibration with human raters at small scale
  - Robustness across commodities (gold, copper, gas)
  - Extending the entity vocabulary with domain-aware embeddings

## Chapter 6 - Conclusion

**Purpose:** Wrap up. Restate the contributions, summarize the findings, point to future work.

- 6.1 Summary of contributions
- 6.2 Summary of findings
- 6.3 Closing remarks

## Appendices

- A. Haiku extraction prompts (system prompt + tool schema, verbatim)
- B. Calibration article-by-article comparison table (Haiku v2 vs GPT v2)
- C. TFT v2 training hyperparameters
- D. Sample of LLM-rejected articles (illustrating the body_valid vs usable disagreement)
- E. Complete OLS regression tables (all lags, with full p-values and confidence intervals)
