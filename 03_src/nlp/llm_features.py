import anthropic

SYSTEM_PROMPT = """\
You are a financial analyst specializing in WTI crude oil markets. Your task is to \
analyze news articles and extract structured features for use in a quantitative \
liquidity impact model.

First, determine whether the article contains substantive WTI-relevant content \
(usable=true). Set usable=false — and return only that field — for: paywall \
placeholders, Cloudflare blocks, cookie notices, or articles that are clearly not \
about oil markets, macro, or geopolitics relevant to oil.

When usable=true, extract the remaining fields:
- sentiment_score: NET directional impact on WTI price. -1.0 = strongly bearish, \
+1.0 = strongly bullish. Assess the actual content, not the headline tone.
- magnitude: how market-moving is this event. 0.0 = negligible, 1.0 = historic. \
Most articles score 0.1–0.4. Major OPEC cut = 0.9. Geopolitical crisis = 1.0.
- event_type: 1–3 categories ordered by salience from: geopolitical, supply, \
demand, macro, technical, other. EIA inventory articles classify as supply or \
macro — do not use “inventory”.
- entities: specific named actors central to the article (not incidentally mentioned). \
Include countries, organizations, oil companies, and key officials. Use the names \
exactly as they appear in the article.
- certainty: how confirmed is the information. 0.0 = rumor, 0.5 = analyst \
forecast, 0.9 = official announcement.
- time_horizon: immediate (hours to 1 day), short_term (days to weeks), or \
structural (months-plus or permanent themes such as energy transition).\
"""

EXTRACTION_TOOL = {
    "name": "extract_article_features",
    "description": (
        "Extract structured WTI-relevant features from a news article for use "
        "in a quantitative liquidity impact model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {
                "type": "boolean",
                "description": (
                    "True if the article contains substantive WTI-relevant content. "
                    "False for paywalls, Cloudflare blocks, cookie notices, or articles "
                    "that are clearly not about oil markets, macro, or geopolitics "
                    "relevant to oil."
                ),
            },
            "sentiment_score": {
                "type": "number",
                "description": (
                    "Net directional impact on WTI price. -1.0 = strongly bearish, "
                    "0.0 = neutral, +1.0 = strongly bullish. Omit when usable=false."
                ),
            },
            "magnitude": {
                "type": "number",
                "description": (
                    "Event importance for WTI markets. 0.0 = negligible, 1.0 = market-moving. "
                    "Routine update = 0.1, major OPEC cut = 0.9, geopolitical crisis = 1.0. "
                    "Omit when usable=false."
                ),
            },
            "event_type": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["geopolitical", "supply", "demand", "macro", "technical", "other"],
                },
                "minItems": 1,
                "maxItems": 3,
                "description": (
                    "1–3 categories ordered by salience. geopolitical = sanctions/conflict/diplomacy, "
                    "supply = OPEC/production/pipelines, demand = consumption/imports, "
                    "macro = Fed/dollar/inflation, technical = price levels/chart patterns, "
                    "other = company earnings/unclassified. Omit when usable=false."
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific named actors central to the article — countries, organizations, "
                    "oil companies, key officials. Use the names exactly as they appear in the "
                    "article. Omit when usable=false."
                ),
            },
            "certainty": {
                "type": "number",
                "description": (
                    "Confidence in reported information. 0.0 = rumor/speculation, "
                    "0.5 = analyst forecast, 0.9 = confirmed fact. Omit when usable=false."
                ),
            },
            "time_horizon": {
                "type": "string",
                "enum": ["immediate", "short_term", "structural"],
                "description": (
                    "Temporal relevance. immediate = hours to 1 day, short_term = days to weeks, "
                    "structural = months-plus or permanent themes (energy transition, "
                    "OPEC long-run policy). Omit when usable=false."
                ),
            },
        },
        "required": ["usable"],
    },
}

_OPTIONAL_FIELDS = [
    "sentiment_score", "magnitude", "event_type",
    "entities", "certainty", "time_horizon",
]


def extract_features(title: str, body: str | None, client: anthropic.Anthropic) -> dict:
    title = str(title).encode("utf-8", errors="ignore").decode("utf-8").strip()
    body_text = ""
    if isinstance(body, str):
        body_text = body.encode("utf-8", errors="ignore").decode("utf-8")[:1500].strip()

    user_text = f"Title: {title}\n\nBody: {body_text}" if body_text else f"Title: {title}"

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_article_features"},
        messages=[{"role": "user", "content": user_text}],
    )

    tool_input = next(
        block.input
        for block in response.content
        if block.type == "tool_use"
    )

    result = {"usable": tool_input["usable"]}
    if result["usable"]:
        for field in _OPTIONAL_FIELDS:
            result[field] = tool_input.get(field, None)
    else:
        for field in _OPTIONAL_FIELDS:
            result[field] = None

    return result
