# HuggingFace Models

## Sentiment analysis models (NLP)

- **[ProsusAI/finBERT](https://huggingface.co/ProsusAI/finbert)**
    - Input: Text like news
    - Output: Percentages on negative, neutral and positive sentiment score.
    - Description: Analyze the sentiment of financial text. the score after processing with ChatGPT changes
    
    FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification. 
    

- **[nickmuchi/sec-bert-finetuned-finance-classification](https://huggingface.co/nickmuchi/sec-bert-finetuned-finance-classification)**
    - Input: Text like news
    - Output: Percenetages on bearish, bullish or neutral score.
    - Description: 


## Prompt to analyze financial text and produce new scores

The experiment gets the text and scores output from finBERT and using also current news and hidden relationships that finBERT its ignoring therefore its scores could not represent accurately what the news impact could be.

```console
I am conducting an experiment to study financial impacts on commodities. The goal is to predict how a given news item affects the liquidity of a commodity. The commodity is not restricted and may vary per news item.

I use a financial sentiment model that scores news paragraphs as negative, neutral, or positive. This model is trained on financial language and terminology.

The core research question is whether a sentiment model, without full real-world context, may classify news as neutral even when the actual market impact is positive or negative. Therefore, sentiment scores alone may be insufficient to predict liquidity behavior.

You will act as a context-aware reasoning agent.

Input you will receive
	1.	A news paragraph
	2.	The sentiment scores produced by the financial model

Your task
	•	Analyze the news using real-world and market context
	•	Evaluate whether the sentiment scores are appropriate for predicting liquidity impact
	•	Adjust the sentiment scores if necessary
	•	Produce a final liquidity impact verdict

Output Requirements (STRICT)

You must output 4 separate code cells, in the following order and format.

Code cell 1 — Preprocessed scores

Negative: xx.yy%
Neutral: xx.yy%
Positive: xx.yy%

Code cell 2 — Processed scores

Negative: xx.yy%
Neutral: xx.yy%
Positive: xx.yy%

Code cell 3 — Veredict

0.xx

Code cell 4 — Justification


Important formatting rules
Each section must be in its own code cell
• Code cells must contain only the values, no explanations
Do not include any additional text outside the code cells

Verdict Definition
Verdict represents the expected liquidity impact of the news on the commodity
It must be a decimal between -1.00 and 1.00
Negative values → expected negative liquidity impact
Positive values → expected positive liquidity impact
Examples:
0.18 → +18% liquidity impact
-0.42 → −42% liquidity impact

The verdict reflects market liquidity impact, not sentiment polarity.

Justification
Why these numbers, justify key parts of the news that define your scores.
Up to date data that aids this decision.
```


## Trade visualizators


- **[Trademap](trademap.org)**
    - Uses the information from comtradeplus.un.org

- **[Comtradeplus (UN page)](comtradeplus.un.org)**
    - More raw information that could be difficult to visualize, powerful

- **[FAOStat](fao.org)**
    - Production values mainly
