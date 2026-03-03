# Project Plan v2.0
## Modeling News-Driven Liquidity Dynamics and Information Propagation in Commodities Markets
**Author:** Enrique Favila Martínez  
**Updated:** March 2026  
**Remaining time:** ~5 months

---

## 1. Updated Project Architecture

```
proyecto_wti/
│
├── 01_data/
│   ├── raw/
│   │   ├── price/                    # Raw OHLCV data from yfinance
│   │   ├── news/                     # Raw news text (GDELT, EIA)
│   │   └── macro/                    # EIA inventories, FRED exchange rates
│   │
│   ├── processed/                    # Cleaned and aligned data
│   └── features/                     # Model-ready features
│
├── 02_notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_prototyping.ipynb
│   ├── 03_event_windows.ipynb        # ✅ Done — EIA event study
│   ├── 04_baseline_models.ipynb      # ✅ Done — OLS baseline
│   └── 05_finbert_prototyping.ipynb  # Next step
│
├── 03_src/
│   ├── adquisicion/
│   │   ├── yfinance_client.py        # ✅ Primary price source (CL=F)
│   │   ├── eia_downloader.py         # ✅ Weekly inventory reports
│   │   ├── gdelt_client.py           # 🔜 Free-text news source
│   │   └── fred_downloader.py        # USD exchange rate (optional)
│   │
│   ├── features/
│   │   ├── build_liquidity_features.py   # ✅ Volume, range, Amihud
│   │   ├── build_news_features.py        # 🔜 FinBERT embeddings
│   │   └── build_event_windows.py        # ✅ ±4h windows around EIA
│   │
│   ├── modelos/
│   │   ├── baseline/
│   │   │   └── ols_baseline.py           # ✅ OLS asymmetry (p=0.03)
│   │   │
│   │   ├── asymmetry/                    # RQ1
│   │   │   ├── neural_asymmetry.py       # Neural net with attention
│   │   │   └── finbert_asymmetry.py      # FinBERT + liquidity model
│   │   │
│   │   ├── temporal/                     # RQ2
│   │   │   ├── var_model.py              # VAR baseline
│   │   │   ├── lstm_model.py             # LSTM temporal model
│   │   │   └── transformer_model.py      # Temporal Fusion Transformer
│   │   │
│   │   └── evaluation/
│   │       ├── metrics.py
│   │       ├── plots.py
│   │       └── robustness.py             # Robustness checks
│   │
│   ├── nlp/                              # NEW — NLP pipeline isolated
│   │   ├── preprocess.py                 # spaCy / NLTK text cleaning
│   │   ├── finbert_embeddings.py         # FinBERT sentiment + embeddings
│   │   └── sentiment_comparison.py       # Compare FinBERT vs SEC-BERT
│   │
│   └── config/
│       ├── config_hourly.yaml            # Primary config (hourly data)
│       └── config_daily.yaml             # Fallback config
│
├── 04_outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/                           # Saved trained models
│   └── experiment_tracking/             # Weights & Biases logs
│
├── 05_reports/
│   ├── month1/
│   │   ├── progress_report_month1.md     # ✅ Done (EN)
│   │   └── informe_mes1_progreso.md      # ✅ Done (ES)
│   ├── month2/
│   ├── literature/
│   │   └── literature_review.md
│   └── thesis/                           # Thesis document (start early)
│       └── thesis_draft.md
│
├── requirements.txt
└── README.md
```

---

## 2. Technology Stack (Updated)

### Core data
| Tool | Purpose | Status |
|---|---|---|
| `yfinance` | WTI hourly OHLCV (CL=F) | ✅ Adopted |
| `EIA API` | Weekly inventory events | ✅ Adopted |
| `GDELT Project` | Free-text news (geopolitical, energy) | 🔜 Next step |
| `FRED API` | USD exchange rate (macro control) | Optional |

### NLP
| Tool | Purpose | Status |
|---|---|---|
| `FinBERT` (HuggingFace) | Financial sentiment + embeddings | 🔜 Phase 2 |
| `SEC-BERT` (HuggingFace) | Alternative sentiment model | 🔜 Phase 2 |
| `spaCy` | Text preprocessing, tokenization | 🔜 Phase 2 |
| `NLTK` | Text cleaning, stopword removal | 🔜 Phase 2 |
| `Sentence-BERT` | Full-article embeddings | 🔜 Phase 3 |

### Modeling
| Tool | Purpose | Status |
|---|---|---|
| `statsmodels` | OLS baseline, VAR | ✅ In use |
| `PyTorch` | Neural networks, LSTM, Transformer | 🔜 Phase 3-4 |
| `HuggingFace Transformers` | FinBERT, fine-tuning | 🔜 Phase 2-3 |
| `scikit-learn` | Preprocessing, evaluation metrics | 🔜 Phase 3 |

### Experiment tracking & reproducibility
| Tool | Purpose | Status |
|---|---|---|
| `Weights & Biases (wandb)` | Experiment tracking, hyperparameter logs | 🔜 Phase 3 |
| `SHAP` | Model interpretability | 🔜 Phase 4 |
| `matplotlib / seaborn` | Visualization | ✅ In use |

---

## 3. Updated Phase Plan

### ✅ Etapa 0 — Completed (Month 1)
- WTI selected as primary commodity
- yfinance adopted as price data source (11,205 hourly records)
- EIA weekly reports adopted as structured news events (321 records)
- Liquidity variables defined: log-volume (primary), price range, Amihud ratio
- Event windows constructed: 101 events, 993 records
- OLS baseline completed: bearish events generate ~23% more volume (p=0.03)

---

### 🔄 Phase 1 — Close out this week
**Goal:** Resolve free-text news source. Close Phase 1.

| Task | Tool | Priority |
|---|---|---|
| Explore GDELT for WTI-related news | GDELT API / BigQuery | 🔴 Critical |
| Verify news coverage 2024–2026 | Python `requests` | 🔴 Critical |
| Align news timestamps with EIA events | `pandas` | 🟡 This week |
| Document Phase 1 decisions | Markdown | 🟡 This week |

**Exit criteria:** At least one free-text news source confirmed with >500 articles covering 2024–2026.

---

### 📋 Phase 2 — Data Acquisition & Feature Engineering (5 weeks)
**Goal:** Complete full data pipeline. NLP baseline running.

**Week 1–2 — GDELT integration + text cleaning**
- Download and filter GDELT articles mentioning WTI / crude oil
- Clean text with spaCy (tokenization, stopword removal, lemmatization)
- Align article timestamps with yfinance hourly data

**Week 3 — FinBERT integration**
- Run FinBERT on cleaned news articles
- Extract: sentiment score (positive/neutral/negative), confidence, embeddings
- Compare FinBERT vs SEC-BERT on same articles (document divergence)

**Week 4 — Feature engineering**
- Aggregate sentiment scores per hour (mean, max, weighted)
- Merge NLP features with liquidity features on hourly timestamps
- Build final modeling dataset: [hour, log_volume, price_range, amihud, sentiment_score, news_direction, hours_from_event]

**Week 5 — Validation & pipeline consolidation**
- Validate dataset completeness and alignment
- Run descriptive statistics on merged dataset
- Begin writing thesis Chapter 3 (Methodology)

**Exit criteria:** Clean merged dataset with price, liquidity, and NLP features ready for modeling.

---

### 📋 Phase 3 — Asymmetry Modeling — RQ1 (4 weeks)
**Goal:** Replace OLS baseline with AI model. Answer RQ1 formally.

**Week 1–2 — Neural asymmetry model**
- Build neural network with separate branches for bearish/bullish sentiment
- Input: FinBERT embeddings + inventory shock features
- Output: predicted log-volume
- Compare against OLS baseline (does AI improve R²?)

**Week 3 — Attention mechanism**
- Add attention layer over sentiment features
- Extract attention weights → which words/phrases drive liquidity response?
- This is the interpretability component for RQ1

**Week 4 — Evaluation & robustness**
- SHAP values for feature importance
- Robustness checks: different time periods, different liquidity metrics
- Write thesis Chapter 4 (Results — RQ1)

**Exit criteria:** Neural model outperforms OLS baseline. Attention weights interpretable. RQ1 answered.

---

### 📋 Phase 4 — Temporal Modeling — RQ2 (5 weeks)
**Goal:** Model lag structure of news impact on liquidity. Answer RQ2.

**Week 1 — VAR baseline**
- Vector Autoregression with sentiment and liquidity variables
- Granger causality tests: does sentiment Granger-cause volume?
- Impulse response functions: how long does a news shock persist?

**Week 2–3 — LSTM model**
- Sequence model: input = [sentiment_t, volume_t-1, ..., volume_t-n]
- Output = predicted volume_t
- Capture non-linear temporal dependencies

**Week 4–5 — Temporal Fusion Transformer**
- More powerful temporal model with interpretable attention over time
- Identifies which past hours are most relevant for predicting current liquidity
- Compare against VAR and LSTM
- Write thesis Chapter 4 (Results — RQ2)

**Exit criteria:** Clear lag structure identified. RQ2 answered with at least two models.

---

### 📋 Phase 5 — Writing & Optional Extension (5 weeks)
**Goal:** Deliver complete thesis. RQ3 if time allows.

**Week 1–2 — Thesis writing (core chapters)**
- Chapter 1: Introduction (refine from proposal)
- Chapter 2: Literature Review
- Chapter 3: Methodology (already started in parallel)
- Chapter 4: Results (RQ1 + RQ2)
- Chapter 5: Discussion & Conclusions

**Week 3 — RQ3 (optional — spillovers)**
- Only if RQ1 and RQ2 are fully closed
- Download Brent, Natural Gas hourly data (yfinance)
- Run VAR spillover analysis (Diebold-Yilmaz framework)
- No GNNs — too risky at this stage

**Week 4–5 — Final revision & submission**
- Supervisor feedback integration
- Reproducibility check (clean repo, requirements.txt, README)
- Final submission

---

## 4. Risk Register (Updated)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GDELT has insufficient WTI coverage | Medium | High | Fall back to EIA events only + manual news dates (OPEC meetings) |
| FinBERT embeddings don't improve over OLS | Medium | Medium | Document as finding — motivates future work |
| Phase 2 takes longer than 5 weeks | High | Medium | Cut Phase 4 to VAR + LSTM only (drop Transformer) |
| No significant results for RQ2 | Low | High | Granger causality negative result is still publishable |
| Time insufficient for RQ3 | High | Low | RQ3 is optional — thesis is complete without it |

---

## 5. Immediate Next Steps (This Week)

1. Explore GDELT for WTI news coverage → confirm or discard as text source
2. If GDELT confirmed → build `gdelt_client.py` acquisition script
3. If GDELT fails → define fallback (OPEC meeting dates + manual EIA news)
4. Close Phase 1 formally → update `informe_mes1` with GDELT decision
5. Install spaCy and test text preprocessing pipeline on first news batch

