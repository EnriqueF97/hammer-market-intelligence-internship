# Modeling News-Driven Liquidity Dynamics in Commodities Markets

**Author:** Enrique Favila Martínez  
**Program:** MSc Artificial Intelligence & Cybersecurity — Radboud University  
**Host company:** Hammer Market Intelligence  
**Supervisor:** Dr. Lejla Batina  

---

## Project Overview

This thesis investigates how news-driven information affects liquidity dynamics in commodities markets, with a focus on WTI Crude Oil. The project addresses three research questions:

- **RQ1** — Do negative (bearish) news events have a significantly larger and more persistent impact on WTI liquidity than positive (bullish) events?
- **RQ2** — What is the lag structure of news impact on liquidity? (effect at +1h, +2h, +5h? Is there reversal?)
- **RQ3** — How do liquidity shocks propagate across related commodities (WTI, Brent, Natural Gas)?

---

## Repository Structure

```text
project_wti/
├── 01_data/                        # Data directory
│   ├── raw/
│   │   ├── price/                  # Raw OHLCV data from yfinance (CL=F)
│   │   ├── news/                   # Raw news text (GDELT, EIA reports)
│   │   └── macro/                  # EIA inventories, FRED exchange rates
│   ├── processed/                  # Cleaned and aligned data
│   └── features/                   # Model-ready feature datasets
│
├── 02_notebooks/                   # Exploration and prototyping notebooks
│   ├── 01_yfinances_prices.ipynb   # Import OHLCV data from yfinance
│   ├── 02_eia_inventories.ipynb    # Import inventory reports from EIA
│   ├── 03_gdelt_headlines.ipynb    # Get oil related news headers and source
│   ├── 04_gdelt_scrapper.ipynb     # Crawl, process, filter and store news content
│   ├── 05_alignment.ipynb          # Align news with nearest inventory report
│   ├── 06_finbert_sentiment.ipynb  # Financial score on news content
│   ├── 07_headline_bias.ipynb      # Comparison between title and title + body
│
├── 03_src/
│   ├── acquisition/                # Data acquisition scripts
│   ├── features/                   # Feature engineering
│   ├── models/                     # Model implementations (baseline, RQ1, RQ2)
│   ├── nlp/                        # NLP pipeline (FinBERT, preprocessing)
│   └── config/                     # Experiment configuration files
│
├── 04_outputs/
│   ├── figures/                    # Generated plots
│   ├── tables/                     # Result tables
│   ├── models/                     # Saved trained models
│   └── experiment_tracking/        # Weights & Biases logs
│
└── 05_reports/                     # Monthly progress reports and thesis draft
```

---

## Data Sources

| Source | Data | Frequency |
| --- | --- | --- |
| yfinance (`CL=F`) | WTI OHLCV — Open, High, Low, Close, Volume | Hourly |
| EIA API | U.S. commercial crude oil inventories | Weekly |
| GDELT Project | News articles (energy, geopolitics) | Real-time |

---

## Liquidity Variables

| Variable | Definition | Source |
| --- | --- | --- |
| Log Volume | Log-transformed hourly trading volume | yfinance |
| Price Range | High − Low per hour (Parkinson proxy) | yfinance |
| Amihud Ratio | Absolute return / Volume (Amihud, 2002) | Derived |

---

## Preliminary Results (Month 1)

- **Dataset:** 11,205 hourly records (2024–2026), 101 EIA event windows
- **Baseline finding:** Bearish EIA reports generate ~23% more trading volume than bullish reports in the 0–2 hour reaction window (p = 0.03, robust across 3 model specifications)
- **Implication:** Preliminary evidence of liquidity asymmetry supporting RQ1

---

## Tech Stack

- **Data:** `yfinance`, `requests`, `pandas`
- **NLP:** `transformers` (FinBERT), `spaCy`, `NLTK`
- **Modeling:** `statsmodels`, `PyTorch`, `scikit-learn`
- **Interpretability:** `SHAP`, attention weights
- **Tracking:** `Weights & Biases`
- **Visualization:** `matplotlib`, `seaborn`

---

## Progress

| Phase | Status | Description |
| --- | --- | --- |
| Phase 0 | Complete | Data sources, liquidity variables, OLS baseline |
| Phase 1 | In progress | Free-text news source (GDELT) |
| Phase 2 | Pending | NLP pipeline, feature engineering |
| Phase 3 | Pending | Asymmetry modeling (RQ1) |
| Phase 4 | Pending | Temporal modeling (RQ2) |
| Phase 5 | Pending | Thesis writing, optional RQ3 |

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns. *Journal of Financial Markets*.
- Araci, D. (2019). FinBERT. *ACL Workshop on Financial Technology and NLP*.
- Parkinson, M. (1980). The extreme value method for estimating variance. *Journal of Business*.
- Smales, L. A. (2015). Asymmetric volatility response to news. *Journal of International Financial Markets*.
- Tetlock, P. C. (2007). Giving content to investor sentiment. *The Journal of Finance*.
