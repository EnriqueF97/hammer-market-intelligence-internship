**Thesis Plan**

# Liquidity Modeling in WTI with AI: Asymmetry and Temporality in News Impact

**Author:** Enrique Favila Martínez  
Institution: Radboud University / Hammer Market Intelligence  
**Supervisor:** Dr. Lejla Batina  
**Estimated Duration:** 5 months  
**Selected Commodity:** WTI Crude Oil (CL)  
**Data Source:** OilPriceAPI (intraday) + CME (Open Interest)

---

## 1. Context and Motivation

This research project aims to study how news affects liquidity in commodity markets, with a focus on:

1. **Asymmetry:** Do negative news events affect liquidity differently than positive news?

2. **Temporality:** How long does the market take to react? How persistent is the effect?

WTI (West Texas Intermediate) is used due to its high volatility in response to geopolitical events, high news frequency, availability of intraday data via OilPriceAPI, and abundant reference literature.

The project is positioned at the intersection of:

- **Market Microstructure:** Study of liquidity (bid-ask spread, depth, open interest).

- **Natural Language Processing (NLP):** Semantic representation of news.

- **Temporal AI Models:** Capturing lagged and nonlinear effects.

---

## 2. Research Questions

**RQ1 (Asymmetry):**  
Do negative (vs. positive) news events have a larger and more persistent impact on WTI liquidity metrics?

**RQ2 (Temporality):**  
What is the lag structure of news impact on liquidity? Are there overreaction and reversal effects?

**RQ3 (Extension – if time permits):**  
How do liquidity shocks propagate across related commodities (e.g., WTI, Brent, Gasoline)?

---

## 3. Study Variables

| Type | Variable | Description | Frequency | Source |
|------|----------|------------|------------|---------|
| Dependent (Y) | Bid-Ask Spread | Difference between best bid and best ask | Minute-by-minute | OilPriceAPI (derived from prices) |
|  | Market Depth | Volume at the top levels of the order book | Minute-by-minute | OilPriceAPI (if available) |
|  | Open Interest | Number of outstanding contracts | Daily | CME (web scraping) |
| Independent (X) | News Representations | Embeddings (FinBERT), lexical sentiment, keywords | Per event | Hammer / free APIs |
| Control | Intraday Volatility | Price range within rolling windows | Minute-by-minute | Calculated from OilPriceAPI |

---

## 4. Phase Plan (5 Months)

### Phase 1: Foundations and Design (2 weeks)

- Guided literature review (Tetlock, Smales, Diebold-Yilmaz, Borovkova).
- Final refinement of RQs.
- Familiarization with OilPriceAPI (free plan).
- Definition of exact liquidity metrics based on available data.

---

### Phase 2: Data Acquisition and Feature Engineering (4 weeks)

- Implement `oilpriceapi_client.py` to download WTI intraday data.
- Implement `cme_scraper.py` for daily Open Interest.
- Implement `build_liquidity_features.py`: computation of spread, depth, cumulative volume.
- Implement `build_news_features.py` with at least:
  - Baseline: Loughran-McDonald lexicon.
  - Advanced: FinBERT embeddings.
- Implement `build_event_windows.py` to extract [-30, +60] minute windows around each news event.

---

### Phase 3: Asymmetry Modeling (RQ1) – 4 weeks

- Linear model with interaction terms (regression).
- Neural network model with separate branches for positive/negative sentiment.
- Comparative evaluation.
- Results visualization.

---

### Phase 4: Temporal Modeling (RQ2) – 5 weeks

- VAR model with impulse-response functions.
- LSTM model with attention.
- (Optional) Temporal Transformer.
- Persistence and reversal analysis.

---

### Phase 5: Extension and Writing (5 weeks)

- (If time permits) RQ3: Spillovers with Graph Neural Networks.
- Thesis writing: Introduction, Related Work, Methodology, Results, Discussion.
- Preparation of a simple dashboard for Hammer.

---

## 5. AI Techniques by Phase

| Phase | AI Techniques | Purpose |
|-------|--------------|----------|
| Phase 2 (NLP) | FinBERT, Sentence-BERT, contextual embeddings | Semantic representation of news |
| Phase 3 (Asymmetry) | Neural networks with separate branches, polarity attention | Capture differential impact of positive/negative news |
| Phase 4 (Temporality) | LSTM, GRU, Transformers, Neural Hawkes Processes | Model temporal evolution of post-news liquidity |
| Phase 5 (Spillovers) | Graph Neural Networks (GNNs) | Model propagation across commodities |
| Interpretability | Attention mechanisms, SHAP, LIME | Explain which words and moments matter |

---

## 6. Risk Management

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| OilPriceAPI lacks bid-ask data | High | Medium | Use intraday volume and range as liquidity proxies |
| NLP too slow/expensive | Medium | Medium | Use lighter Sentence-BERT or precomputed embeddings |
| Models too complex for timeframe | Medium | High | Prioritize simpler but well-executed models (VAR + LSTM) |
| No significant results | Low | High | Keep lexical baseline for comparison; pivot to discrete event analysis |
| Insufficient time for RQ3 | High | Low | RQ3 is “nice to have”; prioritize RQ1 and RQ2 |

---

## 7. Deliverables

1. **Thesis document** including:
   - Introduction and motivation.
   - Literature review.
   - Detailed (reproducible) methodology.
   - Results for RQ1 and RQ2.
   - Discussion and conclusions.

2. **Code repository** (GitHub) with documented modular architecture.

3. **Basic dashboard** for Hammer (optional, if time and resources align).

---

## 8. Visual Timeline
```
Week: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
Phase 1: [X X]
Phase 2: [X X X X]
Phase 3: [X X X X]
Phase 4: [X X X X X]
Phase 5: [X X X X X]
```

## 9. Modular Project Architecture

An architecture is designed to scale from daily to intraday data without rebuilding the project.
```
project_wti/
│
├── 01_data/
│   ├── raw/ # Raw data (downloaded CSV files)
│   ├── processed/ # Cleaned and aligned data
│   └── features/ # Model-ready features
│
├── 02_notebooks/ # Exploration and prototyping
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_prototyping.ipynb
│   └── 03_initial_models.ipynb
│
├── 03_src/ # Modular source code
│   ├── acquisition/
│   │   ├── oilpriceapi_client.py # API client
│   │   ├── cme_scraper.py # Open Interest scraping
│   │   └── eia_downloader.py # Inventory downloads
│   │
│   ├── features/
│   │   ├── build_liquidity_features.py # Computes spread, depth, etc.
│   │   ├── build_news_features.py # NLP invariant (FinBERT, etc.)
│   │   └── build_event_windows.py # Windows around news events
│   │
│   ├── models/
│   │   ├── asymmetry/ # Models for RQ1
│   │   │   ├── linear_interaction.py
│   │   │   └── neural_asymmetry.py
│   │   ├── temporal/ # Models for RQ2
│   │   │   ├── var_model.py
│   │   │   ├── lstm_model.py
│   │   │   └── transformer_model.py
│   │   └── evaluation/ # Metrics and validation
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   └── config/ # Experiment configuration
│   ├── config_daily.yaml
│   └── config_intraday.yaml
│
├── 04_outputs/ # Results, figures, tables
│   ├── figures/
│   ├── tables/
│   └── models/ # Saved trained models
│
├── 05_reports/ # Progress and deliverables
│   ├── literature_review.md
│   ├── phase1_report.md
│   └── ...
│
├── requirements.txt # Dependencies
└── README.md # Project description
```