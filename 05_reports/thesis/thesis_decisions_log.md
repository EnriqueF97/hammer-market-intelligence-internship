# Thesis Decisions Log

A record of design and methodology decisions made during the project, captured to ensure they make it into the thesis writeup. Each entry includes the **decision**, the **reasoning**, and **what to write up in the thesis**.

This document is meant to be a working draft — refine, reorder, and rewrite as the thesis takes shape. Bullets are for capture, not final prose.

---

## Project trajectory (high level)

The project evolved through several methodological pivots, each driven by what the data could and could not support. The narrative of the thesis should make these pivots explicit rather than presenting only the final approach — the reasoning behind the pivots is itself a methodological contribution.

### Commodity selection: sugar → WTI

- **Decision:** Pivoted from sugar to WTI crude oil as the target commodity early in the project.
- **Reasoning:** Sugar showed limited variability and weak geopolitical sensitivity. WTI is meaningfully more responsive to news shocks, has richer event structure (OPEC, geopolitics, EIA inventory releases), and aligns better with the research questions on news-driven liquidity dynamics.
- **Thesis writeup:** A paragraph in the data section justifying commodity choice. Frame the WTI selection as the result of preliminary screening, not as a starting assumption.

### Sentiment extraction: FinBERT → Claude Haiku

- **Decision:** Replaced FinBERT (3-class classification with confidence score) with Claude Haiku LLM extraction of a richer feature schema.
- **Reasoning:**
  - FinBERT's discrete 3-class output (positive/neutral/negative) collapses information that is naturally continuous.
  - FinBERT confidence scores correlate weakly with downstream signal quality.
  - LLM extraction enables structured features beyond sentiment: event type, named entities, certainty, time horizon, magnitude.
  - Richer signal addresses the sparsity problem that ultimately killed the VAR model.
- **Thesis writeup:** Method section needs a comparative justification. Cite FinBERT's role in financial NLP literature, explain its limitations for this specific signal-extraction task, justify the move to a generative LLM with structured outputs. This is a methodological contribution — explicit comparison of two NLP regimes for the same task.

### Modeling approach: lag OLS → VAR → TFT

- **Decision (1):** VAR model was built and abandoned in favor of a Temporal Fusion Transformer.
- **Reasoning:** VAR suffered from signal sparsity (~50% zero-sentiment hours). The estimator struggled to identify dynamics from a target time series dominated by zeros. The problem was not implementation but data structure.
- **Decision (2):** TFT chosen as the primary deep-learning model.
- **Reasoning:**
  - Handles mixed continuous and categorical inputs natively.
  - Variable Selection Networks provide interpretability — feature importance per time step.
  - Multi-head attention exposes which historical hours drove a prediction (directly addresses RQ1 lag structure).
  - Designed for irregular event-driven signal mixed with continuous market data.
- **Decision (3):** Lag OLS retained as the canonical evidence for RQ1 (lag structure of news impact) and RQ2 (bearish vs bullish asymmetry).
- **Reasoning:** TFT's directional asymmetry test was underpowered (p=0.56 on the pre-war model). Lag OLS produces clean, publishable p-values for both RQs and provides a transparent baseline. TFT's role is to (a) confirm patterns at the model level, (b) expose feature importance, (c) test whether richer features capture interactions OLS cannot.
- **Thesis writeup:** Methods chapter should present the analytical stack as a sequence: OLS for hypothesis testing → TFT for nonlinear modeling and interpretability. Frame the choice as complementary rather than as a search for the "best" model.

### Headline bias finding

- **Status:** Methodological contribution, already partly written up.
- **Finding:** Title-only vs title+body inputs to FinBERT produce sentiment disagreement on 41.6% of articles (χ²=2050, p<0.001).
- **Thesis writeup:** This belongs as its own subsection in the methods or results chapter. It motivates the decision to use full article bodies (when available) for LLM feature extraction. Worth a figure (already exists: `headline_bias_comparison.png`).

### Temporal alignment

- **Decision:** Articles are assigned to the next trading hour via ceiling, with forward-assignment for off-hours articles.
- **Reasoning:**
  - Ceiling (rather than rounding) ensures the article's publication time strictly precedes the trading hour it's matched to, eliminating reverse-causal contamination.
  - Forward-assignment (rather than dropping off-hours articles) preserves overnight and weekend news, which traders read at market open and act on. Dropping these would discard exactly the most informative articles (geopolitical events, weekend supply shocks).
- **Thesis writeup:** A methods subsection on temporal alignment. Be explicit about the bias trade-off: ceiling may underestimate intra-hour reactions but eliminates causal leakage. Consider including a robustness check with floor alignment to show the lag-peak finding is not an artifact of the alignment rule.

### Data storage: CSV → SQLite

- **Decision:** Migrated the pipeline from CSV files to a SQLite database (`wti_thesis.db`).
- **Reasoning:**
  - CSVs duplicated state across notebooks, causing drift between intermediate outputs.
  - DB-backed pipeline allows the alignment step to be a simple join rather than a merge of separately-saved files.
  - `datetime_hour` is computed at write time, ensuring all consumers see the same canonical alignment.
- **Thesis writeup:** Brief mention in the methods chapter — primarily an engineering decision, not a methodological one, but worth noting for reproducibility.

---

## Phase 5 decisions (current — May 2026)

This section captures the in-progress decisions for the second LLM extraction pass, which expands and refines the feature pipeline.

### Drop `price_direction` from the feature schema

- **Decision:** Remove `price_direction` from the Haiku extraction. Derive it post-hoc from `sentiment_score` if ever needed.
- **Reasoning:**
  - Empirical analysis of the first 12,024 extractions shows `price_direction` is essentially a discretization of `sentiment_score`. Of 12,024 articles, only 16 (0.13%) showed meaningful disagreement between the two fields.
  - Keeping it costs output tokens and increases the surface area for schema drift (the existing dataset already contains a `slightly_bullish` artifact from prompt non-compliance).
  - A downstream discrete direction feature can be computed deterministically from `sentiment_score` via thresholding.
- **Thesis writeup:** A short methodological note in the feature engineering section. Demonstrates that feature reduction is justified by analysis of pilot extractions, not arbitrary.

### `event_type` as a list

- **Decision:** Change `event_type` from a single string to an array of 1-3 categories ordered by salience.
- **Reasoning:** Articles often span multiple event types ("Iran threatens shipments while OPEC+ debates production cuts ahead of Fed meeting"). Forcing a single category collapses real information. The TFT can consume multi-hot encodings cleanly.
- **Implementation:** DB stores the JSON array as a TEXT column (no schema change needed; `entities` already uses this pattern). The model-prep code multi-hot encodes at dataset construction time, keeping the DB human-readable.
- **Thesis writeup:** Feature engineering section. Explain the tension between human-readable storage and model-ready encoding, and the design choice to handle the transformation in code rather than in the data.

### Remove `inventory` from event_type categories

- **Decision:** `inventory` is no longer a valid `event_type`. Inventory-related articles now fall under `supply` or `macro`.
- **Reasoning:** EIA inventory releases are already a structured signal in the `market_context` table (`eia_surprise`, `is_eia_release`). News articles _about_ inventory data are not a separate methodological category — they're supply news with a structured-data counterpart.
- **Thesis writeup:** Minor — mention in the feature engineering section when listing event categories.

### Refine `time_horizon` to immediate / short_term / structural

- **Decision:** Replace `long_term` with `structural`. Drop the implicit "could be 1 year, could be 2030" ambiguity.
- **Reasoning:** "Long-term" was being applied to two distinct cases: news that affects the next few months (relevant to the model) and news about structural multi-year themes (mostly irrelevant to an hourly liquidity model). `structural` cleanly tags the latter, allowing the model to down-weight it via the Variable Selection Network.
- **Thesis writeup:** Feature definition table or list. Briefly explain the three buckets and why `structural` exists as its own category.

### Entity normalization: post-extraction, in code

- **Decision:** Do not enforce entity name normalization in the prompt. Normalize variants (e.g. "Fed" / "Federal Reserve" / "U.S. Federal Reserve") in a post-extraction Python dictionary.
- **Reasoning:**
  - The LLM is poor at remembering long lists of canonical names.
  - The space of entities is not knowable in advance; new ones appear as news evolves.
  - Post-hoc normalization is easy to iterate on without re-running extractions.
- **Thesis writeup:** Feature engineering section. Mention the normalization step and report the canonical entity list used in the final binary flags.

### Entity encoding: top-N binary flags

- **Decision:** Encode entities as binary flags for the top 15 entities (by frequency) plus a `has_other_entity` catchall, rather than as a single categorical embedding.
- **Reasoning:**
  - Articles routinely mention multiple entities; a single categorical can only represent one.
  - The TFT's Variable Selection Network handles sparse binary features well — it learns to down-weight uninformative flags automatically.
  - Top-15 captures the bulk of signal; the long tail (746 entities appearing only once) is noise.
- **Thesis writeup:** Feature engineering section. Justify the top-N cutoff with the entity frequency distribution.

### Tool-use API with strict JSON schema

- **Decision:** Use Anthropic's tool-use API with `tool_choice` forcing the extraction tool, rather than relying on the LLM to produce valid JSON via prompt instruction.
- **Reasoning:**
  - The first extraction pass produced rare but real schema leakage (`slightly_bullish`, `medium_term`).
  - Tool-use enforces enum constraints, type constraints, and required fields at the API level.
  - Eliminates JSON parsing brittleness in the pipeline.
- **Thesis writeup:** Methods — pipeline architecture. Mention as a quality control measure for LLM-extracted features.

### `usable` flag as the canonical filter

- **Decision:** Add a `usable` boolean to the extraction schema. Send all articles with a real body (regardless of `body_valid`) to the LLM. Let the LLM decide which articles are usable for modeling. `usable` becomes the canonical downstream filter.
- **Reasoning:**
  - Inspection of `body_valid=0` articles revealed legitimate news being rejected by the regex/keyword heuristic (false positives).
  - Inspection of `body_valid=1` articles revealed boilerplate slipping through (false negatives).
  - The LLM is a stronger judge of content usability than a regex heuristic.
  - With Batches API + prompt caching, the marginal cost of sending more articles is small relative to the methodological gain.
- **Implementation:**
  - The `usable` field is the only `required` field in the tool schema. When `usable=false`, the LLM returns only that field; other fields are omitted (saving output tokens and avoiding hallucinated values).
  - Downstream pipelines filter on `WHERE usable = 1`.
  - The old `body_valid` heuristic is retained for comparison but no longer the canonical filter.
- **Pending validation (calibration):**
  - 30 articles will be hand-scored by the author across three strata: `body_valid=1` (10), `body_valid=0` with substantial content (10), `body_valid=0` with weaker content (10).
  - Human scores compared to LLM `usable` flag to estimate agreement and identify systematic biases before committing to the full batch.
- **Pending analysis (post-batch):**
  - Compute disagreement between `body_valid` and `usable` across the full dataset.
  - Quantify false-positive rate (legit articles `body_valid` rejected) and false-negative rate (boilerplate `body_valid` accepted).
  - Compute inter-rater agreement (Cohen's κ) between the regex filter and the LLM filter.
  - Audit a sample of cases where the LLM says `usable=false` but the article appears legitimate to a human — characterize the LLM's failure modes.
- **Thesis writeup:** This is a methodological contribution in its own right. Frame it as: _"Two filtering regimes were compared — a regex/keyword heuristic and an LLM-based usability judgment. The disagreement rate was X%, with the LLM recovering Y articles incorrectly rejected by the heuristic and rejecting Z articles incorrectly accepted by the heuristic. Inter-rater agreement was κ=K, indicating substantial but imperfect overlap. Manual audit of LLM rejections suggests its failure modes are primarily [characterize]."_ This is publishable material — comparison of filtering approaches for noisy financial news corpora is genuinely useful to the literature.

### Cost optimizations

- **Decision:** Use Anthropic Batches API + prompt caching on the system prompt + minimal-output short-circuit for `usable=false`.
- **Reasoning:**
  - Batches API: 50% discount, asynchronous processing (1-24h turnaround), no rate-limit babysitting.
  - Prompt caching: ~700 tokens of system prompt + tool schema are identical across ~20k calls; caching reduces input cost on that portion by 90%.
  - Short-circuit: when `usable=false`, output is ~10 tokens instead of ~200. Saves ~$1-2 on the batch.
- **Estimated total cost:** ~$15-20 for the full batch over ~20k articles.
- **Thesis writeup:** Brief footnote or appendix on reproducibility. State the model version (`claude-haiku-4-5`), temperature, the use of Batches API, and the cost. Reproducibility-relevant, not methodologically central.

---

## Open methodological questions (to revisit before final writeup)

- **TFT directional asymmetry test was underpowered (p=0.56).** Will retraining on the expanded dataset with proper categorical handling and entity flags change this? If still underpowered, frame TFT as primarily addressing RQ1 (lag structure via attention) and feature importance, with lag OLS as the canonical evidence for RQ2 (asymmetry).
- **Daily memory effect at −27/−28h in TFT attention.** Genuinely interesting but needs an interpretation. Is it traders revisiting positions at the same time the next day, overnight cycle effects, or an artifact of the encoder window interacting with `is_us_session`? Worth a paragraph in the discussion but worth thinking through _why_ first.
- **Robustness check for the alignment rule (ceiling vs floor vs round).** A 30-minute experiment that would strengthen the methods chapter. Re-run lag OLS with floor alignment; report whether the lag-peak shifts.
- **Pre-war vs post-war TFT comparison.** The first TFT was trained on pre-war (Feb 2026) data. The retraining will include the post-war expansion. The pre-war model's clean results were the original publishable finding; the post-war model is an extension. Both should be reported, with discussion of how (or whether) the geopolitical shock affected the model.

---

## Maintenance

- Update this document after each significant methodological decision.
- Keep the format: short bullets, no dense prose, decisions traceable to reasoning.
- When the thesis writing phase begins, this document becomes the skeleton of the methods chapter.
