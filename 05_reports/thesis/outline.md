# Thesis Outline

**Title (working):** Modeling News-Driven Liquidity Dynamics and Information Propagation in Commodities Markets

**Author:** Enrique [last name]
**Supervisor:** Dr. Lejla Batina
**Institution:** Radboud University — MSc Artificial Intelligence
**Host:** Hammer Market Intelligence

---

## Purpose of this document

This file is the canonical structural reference for the thesis. Every drafting session opens with this document in context. It defines what goes in each chapter, in what order, and at roughly what length. The detailed content of each section is drafted separately; this document is the skeleton.

The structure follows the convention used at Lejla's lab (cf. Niayesh 2025, Gheorghiu 2025): six chapters, with Background combining context and related work, and Experiments doing most of the heavy lifting.

---

## Conventions

- **Citation style:** IEEE numbered (`[1]`, `[2]`).
- **Voice:** first-person plural ("we") where natural, passive where appropriate. Avoid first-person singular.
- **Tone:** C1–C2 academic English. Content-dense paragraphs; no padding. No em dashes.
- **Figures and tables:** Numbered per chapter (Figure 4.1, Table 3.2). Every figure has a substantive caption.
- **Equations:** Numbered, referenced as Eq. (3.1) etc.
- **Length target:** 48–58 pages of content excluding bibliography, appendices, and front matter.

---

## Chapter 1 — Introduction (~3 pages)

**Purpose:** Position the work. Motivate the problem, state the research questions, list the contributions, preview the structure.

- 1.1 Motivation
  - News shapes commodity markets, but most empirical work operates at daily resolution, missing the intra-day dynamics where information actually propagates
  - WTI crude oil specifically has a rich event structure (geopolitics, OPEC, EIA inventories, macro) that makes it well-suited for hourly-resolution analysis
- 1.2 Problem statement
  - How does news propagate into WTI liquidity at hourly resolution?
  - What is the lag structure, the directional asymmetry, the role of macro and event-type features?
- 1.3 Research questions
  - RQ1: lag structure of news impact on liquidity
  - RQ2: bearish vs bullish asymmetry in liquidity response
  - RQ3 (deferred): cross-commodity spillovers
- 1.4 Contributions
  - (a) Empirical lag-and-asymmetry finding for WTI at hourly resolution, with peak news impact at lag +6h and a consistent bearish > bullish asymmetry. Supported by lag OLS results (§4.2.4) and confirmed by TFT v2 attention patterns (§4.3.7).
  - (b) Methodological contribution: a decomposition of LLM-extracted news features into orthogonal economic channels (supply, demand, risk premium), validated by inter-model calibration. The decomposition improves cross-model agreement on news sentiment by roughly 50 percentage points relative to a single composite score. Supported by §4.3.2–§4.3.5.
  - (c) Comparative finding on news content representation: title-only and full-body sentiment extraction disagree on 41.6% of articles, with systematic asymmetry. Supported by §4.2.2.
- 1.5 Structure of the thesis (one paragraph)

---

## Chapter 2 — Background (~10–12 pages)

**Purpose:** Give the reader everything needed to follow Chapters 3 and 4. Covers terminology and state of the art.

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

---

## Chapter 3 — Methods (~8–10 pages)

**Purpose:** Describe the techniques used in the thesis. Method-centric, not chronological. Each phase in Chapter 4 invokes these techniques.

- 3.1 Data sources and acquisition
  - Commodity selection: WTI crude oil was chosen after preliminary screening of candidate commodities; sugar futures were considered initially but excluded due to limited price variability and weak responsiveness to news events at hourly resolution
  - Market data via yfinance (hourly OHLCV)
  - EIA Weekly Petroleum Status Reports
  - GDELT headline scraping (eight thematic queries)
  - Body retrieval via BeautifulSoup
- 3.2 Temporal alignment
  - Ceiling assignment to next trading hour (causality preservation)
  - Forward-assignment of off-hours articles to the next available trading hour
  - The assignment_gap diagnostic and contemporaneous vs forward-assigned categorization
- 3.3 News feature extraction approaches
  - 3.3.1 FinBERT: 3-class sentiment + confidence
  - 3.3.2 Structured LLM extraction with Claude Haiku and tool-use API
  - 3.3.3 Channel decomposition (sentiment, supply, demand, risk_premium)
  - Schema design choices and orthogonality rationale
- 3.4 Filtering strategies
  - 3.4.1 Regex/keyword heuristic (body_valid)
  - 3.4.2 LLM-judged usable flag
- 3.5 Statistical models
  - 3.5.1 Lag OLS specification
  - 3.5.2 Vector Autoregression specification
- 3.6 Deep learning model: Temporal Fusion Transformer
  - Architecture summary
  - Variable selection network
  - Multi-head attention
  - Quantile loss and prediction intervals
- 3.7 Inter-model calibration methodology
  - Stratified sampling procedure
  - Agreement metrics (Pearson correlation, sign agreement, MAD, Cohen's kappa for filters)
  - Limitations of inter-LLM calibration as a substitute for human ground truth

---

## Chapter 4 — Experiments (~22–28 pages, the main chapter)

**Purpose:** Document what was done, what was found, and how each finding motivated the next step. Organized in two methodological phases.

### 4.1 Experimental setup

- Hardware (Colab for TFT training, local for pipeline)
- Software stack (Python 3.13, key libraries)
- Data partition strategy: temporal holdout (last 20% as validation)
- Reproducibility: seeds, version-locked dependencies, deterministic prompts

### 4.2 Phase 1 — Initial pipeline and baseline modeling

**Phase scope:** Pre-war data (Jan 2024 → Feb 2026), FinBERT sentiment extraction, regex-based filter, contemporaneous + lag OLS + VAR + TFT v1. The phase establishes the empirical findings on RQ1 and RQ2 and exposes the limitations that motivate Phase 2.

- 4.2.1 Phase 1 data and feature setup
  - Coverage period, GDELT scrape (~13,690 articles), FinBERT extraction
  - EIA inventory features (eia_surprise, is_eia_release)
  - body_valid regex filter
  - 4.2.1 includes a table summarizing the Phase 1 dataset
- 4.2.2 Headline bias experiment
  - Methodology: FinBERT applied to title-only and to title+body, same model, same articles
  - Result: 41.6% sentiment disagreement, χ²=2050, p<0.001
  - Asymmetry: titles overrepresent negative sentiment
  - **Figure: headline_bias_comparison.png**
  - Implication: full bodies preferred for downstream feature extraction
- 4.2.3 Contemporaneous OLS
  - Methodology: news sentiment at hour t regressed on log_volume at hour t
  - Result: R² < 0.001, neither bearish nor bullish significant
  - Interpretation: news does not move volume instantaneously at the moment of publication
  - This null result motivates the lag analysis
- 4.2.4 Lag OLS — RQ1 and RQ2
  - Methodology: log_volume at t+k regressed on bearish/bullish indicators at t, for k ∈ {1, 2, 3, 4, 6, 8, 12}
  - Result: peak impact at lag +6h, consistent bearish > bullish asymmetry
  - Statistical significance (p<0.05) at lags 1, 3 (bullish only), 4, 6
  - R² ≤ 0.002 across lags — sentiment as one signal among many
  - **Figure: lag_coefficients.png**
  - Canonical evidence for RQ1 and RQ2
- 4.2.5 Vector Autoregression
  - Methodology: VAR on hourly time series of sentiment and liquidity
  - Result: impulse response function suggests modest persistent effect but with wide error bands
  - **Figure: irf_sentiment_to_volume.png**
  - Abandonment rationale: ~50% zero-sentiment hours (signal sparsity), VAR estimator underidentified, IRF confidence intervals span zero at most horizons
  - This motivates a model that handles sparse event-driven signal natively → TFT
- 4.2.6 Temporal Fusion Transformer v1
  - Architecture and training setup
  - Feature set: log_volume, price_range, log_return, amihud, FinBERT sentiment_score and confidence, hour, day_of_week, month, is_wednesday, is_us_session
  - Result: val_loss 0.209, attention peak at −4h (consistent with OLS lag +6h finding)
  - Feature importance: sentiment_score at 53%
  - "Daily memory" finding at −27/−28h
  - Limitations:
    - Directional asymmetry test underpowered (p=0.56)
    - event_type integer-encoded as continuous (no real signal)
    - No entity-level awareness
    - Pre-war data only
    - FinBERT signal is coarse (3-class + confidence)
- 4.2.7 Phase 1 summary
  - Findings: lag +6h peak, bearish > bullish asymmetry, headline bias, baseline TFT performance
  - Limitations driving Phase 2:
    - FinBERT cannot represent event type, entities, or causal direction
    - Regex filter has documented false positives
    - Pre-war data misses the Iran-war regime
    - Models lack macro covariates (DXY, VIX)
    - TFT v1 cannot disambiguate event-driven vs price-driven sentiment

### 4.3 Phase 2 — Refined pipeline and improved modeling

**Phase scope:** Post-war data extension, macro covariates added, Claude Haiku structured LLM extraction, channel decomposition, LLM-judged usable filter, inter-model calibration, TFT v2 with proper categorical and entity encodings. The phase responds to Phase 1 limitations and answers the research questions with stronger evidence.

- 4.3.1 Phase 2 data and feature setup
  - GDELT scrape extension through May 2026 (post-war period)
  - Full dataset: 22,795 articles
  - New macro features: DXY, VIX (rationale: news features alone explained <0.2% of volume variance in Phase 1)
  - Table summarizing the Phase 2 dataset, with delta-from-Phase-1 columns
- 4.3.2 LLM feature extraction with Claude Haiku
  - Migration rationale (FinBERT 3-class output collapses information; confidence scores noisy)
  - Tool-use API with strict JSON schema (no parsing brittleness)
  - Initial schema: sentiment_score, magnitude, event_type, entities, certainty, price_direction, time_horizon
  - Prompt design and iteration
  - Cost and operational details (Batches API, prompt caching)
- 4.3.3 LLM usable flag and filter comparison
  - Replacing the regex heuristic with an LLM judgment of content usability
  - Comparison: body_valid (13,550 accepted) vs usable (11,675 accepted), Cohen's κ
  - False-positive cases (regex rejected, LLM accepted): legitimate analyst commentary
  - False-negative cases (regex accepted, LLM rejected): boilerplate slipping through
  - Methodological contribution: LLM-based filtering for noisy news corpora
- 4.3.4 Inter-model calibration of LLM features
  - Methodology: 30-article stratified sample, scored by Haiku and by a GPT-family reference model with the identical prompt
  - First calibration result: sentiment_score correlation 0.39 between models, sign disagreement on 4/13 usable articles (31%)
  - Diagnostic: disagreements concentrate on high-magnitude geopolitical events
  - Root cause analysis: sentiment_score conflates event valence with directional price impact
- 4.3.5 Channel decomposition response
  - Schema extension: supply_impact, demand_impact, risk_premium, each on [-1, +1]
  - Each channel is a single factual judgment, designed to be orthogonal
  - sentiment_score retained for continuity with the headline bias experiment and as a composite
  - price_direction dropped (empirically redundant with sentiment_score; only 16/12024 cases meaningfully differed)
  - event_type changed from single string to array of 1–3 categories
  - Second calibration result:
    - supply_impact: r=0.94, MAD 0.11
    - demand_impact: r=0.96, MAD 0.05
    - risk_premium: r=0.82, MAD 0.13
    - sentiment_score (as side effect): r=0.39 → r=0.88
    - Sign disagreements: 4/13 → 1/14
  - Within-model orthogonality preserved (all pairwise |r| < 0.5)
- 4.3.6 Full LLM extraction batch
  - Execution: 19,619 articles via Batches API, Haiku 4.5
  - Cost: actual ≈ $X (low single digits)
  - Outcome: 11,675 usable (59.5%), 7,944 unusable, ~10% all-channel-zero rate (matching calibration target)
  - Population statistics: channel means show risk premium tilt consistent with the Iran-war period
- 4.3.7 Temporal Fusion Transformer v2
  - Architecture changes from v1:
    - Proper categorical encodings (event_type, time_horizon)
    - Multi-hot entity binary flags (top 15 entities)
    - Three orthogonal channel features
    - Macro covariates (DXY, VIX)
  - Training on the full 22,795-aligned dataset
  - Train/val split: temporal holdout
  - Results: val_loss, attention patterns, feature importance per channel
  - Comparison to v1: where Phase 2 features improved or did not improve the model
- 4.3.8 Phase 1 vs Phase 2 comparison
  - Table: val_loss, top-3 features by importance, asymmetry test p-value
  - Discussion of which Phase 2 changes contributed most to which improvements

### 4.4 Robustness checks

- 4.4.1 Filter comparison (body_valid vs usable, with the κ analysis from 4.3.3)
- 4.4.2 Optional: alignment robustness (lag analysis re-run with floor alignment instead of ceiling, to verify the lag +6h finding is not an artifact of the alignment rule)

---

## Chapter 5 — Discussion (~3–4 pages)

**Purpose:** Interpret the findings, contextualize them in the literature, identify limitations.

- 5.1 Findings on RQ1 (lag structure)
  - The lag +6h peak in OLS, confirmed by TFT attention at −4h
  - What this means about how news information propagates
- 5.2 Findings on RQ2 (asymmetry)
  - Bearish > bullish in OLS is statistically robust
  - TFT directional asymmetry test underpowered; lag OLS retains canonical evidence
  - Methodological note on why this happened (TFT operates on log_volume predictions, not on lag-structured impulse responses)
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
  - RQ3 cross-commodity spillovers (deferred from this thesis)
  - Larger calibration with human raters at small scale + LLM scale
  - Robustness across commodities (gold, copper, gas)
  - Extending the entity vocabulary with domain-aware embeddings

---

## Chapter 6 — Conclusion (~2 pages)

**Purpose:** Wrap up. Restate the contributions, summarize the findings, point to future work.

- 6.1 Summary of contributions
- 6.2 Summary of findings
- 6.3 Closing remarks

---

## Appendices (estimated 5–10 pages, do not count toward content target)

- A. Haiku extraction prompts (system prompt + tool schema, verbatim)
- B. Calibration article-by-article comparison table (Haiku v2 vs GPT v2)
- C. TFT v2 training hyperparameters
- D. Sample of LLM-rejected articles (illustrating the body_valid vs usable disagreement)
- E. Complete OLS regression tables (all lags, with full p-values and confidence intervals)

---

## Drafting order

Recommended sequence for writing:

1. Chapter 3 (Methods) — anchors everything else
2. Chapter 4 Phase 1 (§4.2) — work you know cold, figures exist
3. Chapter 4 Phase 2 except TFT v2 (§4.3.1 through §4.3.6) — recently done, also fresh
4. Chapter 4 TFT v2 results (§4.3.7) and Phase 1 vs Phase 2 comparison (§4.3.8) — after training session
5. Chapter 2 (Background) — done after methodology is locked
6. Chapter 1 (Introduction) — easier once you know what the thesis claims
7. Chapter 5 (Discussion) — late, synthesizing
8. Chapter 6 (Conclusion) — last

---

## Maintenance

- Update this document if structural changes are agreed in chat (e.g., merging or splitting subsections).
- Do not let chapter section structures drift between this file and the actual chapter drafts; the file is the source of truth.
- Add appendices as needed during drafting.
