import json

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
- supply_impact: Direction of impact on WTI supply. Score the article's implication \
for the quantity of oil available to the market.
  +1.0 = strong supply increase (OPEC adds barrels, sanctions lifted on a major \
producer, new pipeline online, Venezuela ramping up).
  +0.5 = moderate supply increase (single-country production gains, inventory builds).
  0.0 = no supply implication.
  -0.5 = moderate supply decrease (refinery outages, minor OPEC cuts).
  -1.0 = strong supply decrease (major OPEC cuts, infrastructure attacks, embargoes, \
refinery fires affecting multiple facilities).
- demand_impact: Direction of impact on WTI demand. Score the article's implication \
for oil consumption.
  +1.0 = strong demand increase (China stimulus, Fed pivot to easing, growth surprise, \
summer driving acceleration).
  +0.5 = moderate demand increase (regional growth indicators).
  0.0 = no demand implication.
  -0.5 = moderate demand decrease (mild slowdown signals).
  -1.0 = strong demand decrease (recession confirmed, Fed strongly hawkish, major \
economy lockdowns, demand destruction events).
- risk_premium: Change in geopolitical or operational risk premium for oil. Score the \
direction of risk change, not the absolute level.
  +1.0 = major escalation (military strike on oil infrastructure, war, Strait of \
Hormuz incident, major sanctions imposed).
  +0.5 = moderate escalation (rising tensions, threats, troop movements).
  0.0 = no change in risk environment.
  -0.5 = moderate de-escalation (talks resuming, partial sanctions lifted).
  -1.0 = major de-escalation (ceasefire signed, diplomatic breakthrough, full \
sanctions removal).
- magnitude: how market-moving is this event. 0.0 = negligible, 1.0 = historic. \
Most articles score 0.1–0.4. Major OPEC cut = 0.9. Geopolitical crisis = 1.0.
- event_type: 1–3 categories ordered by salience from: geopolitical, supply, \
demand, macro, technical, other. EIA inventory articles classify as supply or \
macro — do not use "inventory".
- entities: specific named actors central to the article (not incidentally mentioned). \
Include countries, organizations, oil companies, and key officials. Use the names \
exactly as they appear in the article.
- certainty: how confirmed is the information. 0.0 = rumor, 0.5 = analyst \
forecast, 0.9 = official announcement.
- time_horizon: immediate (hours to 1 day), short_term (days to weeks), or \
structural (months-plus or permanent themes such as energy transition).

The three impact fields (supply_impact, demand_impact, risk_premium) are intentionally \
orthogonal. An article can affect multiple channels independently. Score each based on \
what the article actually says about that specific channel. If the article has no clear \
implication for a channel, score it 0.0.
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
            "supply_impact": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Direction of impact on WTI supply. +1=strong supply increase, -1=strong supply decrease, 0=no implication. See system prompt for anchors. Omit when usable=false.",
            },
            "demand_impact": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Direction of impact on WTI demand. +1=strong demand increase, -1=strong demand decrease, 0=no implication. See system prompt for anchors. Omit when usable=false.",
            },
            "risk_premium": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Change in geopolitical/operational risk premium for oil. +1=major escalation, -1=major de-escalation, 0=no change. See system prompt for anchors. Omit when usable=false.",
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
                    "enum": [
                        "geopolitical",
                        "supply",
                        "demand",
                        "macro",
                        "technical",
                        "other",
                    ],
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
    "sentiment_score",
    "supply_impact",
    "demand_impact",
    "risk_premium",
    "magnitude",
    "event_type",
    "entities",
    "certainty",
    "time_horizon",
]


# ── Entity normalization ───────────────────────────────────────────────────────
# Source: full raw_entity_counts distribution audit on llm_features.entities
#         (Phase 2 corpus, usable=1). Triage thresholds:
#           ≥100 occurrences : reviewed exhaustively
#           50–99            : reviewed exhaustively; 25 occ. min for new canonicals
#           25–49            : reviewed exhaustively; same threshold
#           <25              : dropped unless unambiguous variant of existing canonical
# 70 canonical entities retained after manual review.
# Variants not listed here are silently dropped by canonicalize_entities().

ENTITY_CANONICAL: dict[str, str] = {
    # --- States and political entities ---
    "United States": "US",
    "US": "US",
    "U.S.": "US",
    "United States of America": "US",
    "Iran": "Iran",
    "Tehran": "Iran",
    "Islamic Revolutionary Guard Corps": "Iran",
    "Russia": "Russia",
    "Russian Federation": "Russia",
    "Moscow": "Russia",
    "China": "China",
    "Beijing": "China",
    "Xi Jinping": "China",
    "Israel": "Israel",
    "India": "India",
    "Narendra Modi": "India",
    "Saudi Arabia": "Saudi Arabia",
    "Ukraine": "Ukraine",
    "Venezuela": "Venezuela",
    "Canada": "Canada",
    "Iraq": "Iraq",
    "Nigeria": "Nigeria",
    "United Arab Emirates": "UAE",
    "UAE": "UAE",
    "Abu Dhabi": "UAE",
    "Kazakhstan": "Kazakhstan",
    "Qatar": "Qatar",
    "Oman": "Oman",
    "Japan": "Japan",
    "Kuwait": "Kuwait",
    "Pakistan": "Pakistan",
    "Libya": "Libya",
    "Mexico": "Mexico",
    "Azerbaijan": "Azerbaijan",
    "Yemen": "Yemen",
    "Lebanon": "Lebanon",
    "Brazil": "Brazil",
    "South Korea": "South Korea",
    "Guyana": "Guyana",
    "United Kingdom": "UK",
    "UK": "UK",
    "Britain": "UK",
    "Algeria": "Algeria",
    "Germany": "Germany",
    "Australia": "Australia",
    "Hungary": "Hungary",
    "Egypt": "Egypt",
    "Türkiye": "Türkiye",
    "Turkey": "Türkiye",
    # --- Geographic regions / strategic locations ---
    "Strait of Hormuz": "Strait of Hormuz",
    "Middle East": "Middle East",
    "West Asia": "Middle East",
    "Gaza": "Gaza",
    "Red Sea": "Red Sea",
    "Persian Gulf": "Persian Gulf",
    "Gulf of Mexico": "Gulf of Mexico",
    "Permian Basin": "Permian Basin",
    "Europe": "Europe",
    "Asia": "Asia",
    "Eurozone": "EU",
    # --- Political/economic actors, conflicts and individuals ---
    "Donald Trump": "Trump",
    "Trump": "Trump",
    "President Trump": "Trump",
    "President Donald Trump": "Trump",
    "Trump administration": "Trump",
    "Nicolas Maduro": "Maduro",
    "Nicolás Maduro": "Maduro",
    "Vladimir Putin": "Putin",
    "Jerome Powell": "Powell",
    "Scott Bessent": "Bessent",
    "Joe Biden": "Biden",
    "Biden": "Biden",
    "Biden administration": "Biden",
    "President Biden": "Biden",
    "Fatih Birol": "IEA",
    # --- Israel conflict mapped to Israel because of market moving actor ---
    "Israel-Hamas": "Israel",
    "Israel-Palestine": "Israel",
    "Israeli-Palestinian conflict": "Israel",
    # --- Iranian leadership and senior officials → Iran ---
    "Ebrahim Raisi": "Iran",
    "Raisi": "Iran",
    "Ayatollah Ali Khamenei": "Iran",
    "Ali Khamenei": "Iran",
    "Khamenei": "Iran",
    "Ayatollah Khamenei": "Iran",
    "Masoud Pezeshkian": "Iran",
    "Pezeshkian": "Iran",
    "Hassan Rouhani": "Iran",
    "Rouhani": "Iran",
    "Mahmoud Ahmadinejad": "Iran",
    "Hassan Nasrallah": "Hezbollah",
    "Mohammad Javad Zarif": "Iran",
    "Javad Zarif": "Iran",
    "Hossein Amir-Abdollahian": "Iran",
    "Abbas Araghchi": "Iran",
    "Mohammad Eslami": "Iran",
    "Mohsen Paknejad": "Iran",
    "Javad Owji": "Iran",
    "Mohammad Bagheri": "Iran",
    "Hossein Salami": "Iran",
    "IRGC": "Iran",  # Already mapped in earlier list under "Islamic Revolutionary Guard Corps"; check for this variant
    "Quds Force": "Iran",
    "Ismail Qaani": "Iran",
    # --- Saudi leadership and senior officials → Saudi Arabia ---
    "Mohammed bin Salman": "Saudi Arabia",
    "Mohammad bin Salman": "Saudi Arabia",
    "MBS": "Saudi Arabia",
    "Crown Prince Mohammed bin Salman": "Saudi Arabia",
    "King Salman": "Saudi Arabia",
    "Salman bin Abdulaziz": "Saudi Arabia",
    "Prince Abdulaziz bin Salman": "Saudi Arabia",
    "Abdulaziz bin Salman": "Saudi Arabia",
    "Khalid Al-Falih": "Saudi Arabia",
    "Saudi Energy Ministry": "Saudi Arabia",
    "Saudi Ministry of Energy": "Saudi Arabia",
    "Faisal bin Farhan": "Saudi Arabia",
    "Amin Nasser": "Saudi Aramco",
    # --- Israeli leadership and senior officials → Israel ---
    "Yoav Gallant": "Israel",
    "Israel Katz": "Israel",
    "Itamar Ben-Gvir": "Israel",
    "Bezalel Smotrich": "Israel",
    "Herzi Halevi": "Israel",
    "Ronen Bar": "Israel",
    "Yair Lapid": "Israel",
    "Naftali Bennett": "Israel",
    "Isaac Herzog": "Israel",
    "Benny Gantz": "Israel",
    "Israel Defense Forces": "Israel",
    "IDF": "Israel",
    "Mossad": "Israel",
    "Shin Bet": "Israel",
    # --- Russian leadership beyond Putin/Novak (already mapped) → Russia ---
    "Sergey Lavrov": "Russia",
    "Lavrov": "Russia",
    "Mikhail Mishustin": "Russia",
    "Mishustin": "Russia",
    "Dmitry Medvedev": "Russia",
    "Medvedev": "Russia",
    "Dmitry Peskov": "Russia",
    "Peskov": "Russia",
    "Anton Siluanov": "Russia",
    "Russian Foreign Ministry": "Russia",
    "Kremlin": "Russia",
    "Russian government": "Russia",
    "Bank of Russia": "Russia",
    "Central Bank of Russia": "Russia",
    "Elvira Nabiullina": "Russia",
    "Sergei Shoigu": "Russia",
    "Andrey Belousov": "Russia",
    # --- Multilateral organizations and central banks ---
    "OPEC+": "OPEC+",
    "Opec+": "OPEC+",
    "OPEC Plus": "OPEC+",
    "OPEC": "OPEC",
    "Opec": "OPEC",
    "Organization of the Petroleum Exporting Countries": "OPEC",
    "Federal Reserve": "Fed",
    "Fed": "Fed",
    "US Federal Reserve": "Fed",
    "U.S. Federal Reserve": "Fed",
    "European Union": "EU",
    "EU": "EU",
    "Energy Information Administration": "EIA",
    "EIA": "EIA",
    "U.S. Energy Information Administration": "EIA",
    "US Energy Information Administration": "EIA",
    "International Energy Agency": "IEA",
    "IEA": "IEA",
    "American Petroleum Institute": "API",
    "API": "API",
    # --- Other central banks ---
    "European Central Bank": "ECB",
    "ECB": "ECB",
    "Christine Lagarde": "ECB",
    "Lagarde": "ECB",
    "Bank of England": "UK",
    "BoE": "UK",
    "Andrew Bailey": "UK",
    "Bank of Japan": "Japan",
    "BoJ": "Japan",
    "Kazuo Ueda": "Japan",
    "People's Bank of China": "China",
    "PBOC": "China",
    "PBoC": "China",
    "Reserve Bank of India": "India",
    "RBI": "India",
    "Shaktikanta Das": "India",
    "Bank of Canada": "Canada",
    "BoC": "Canada",
    "Tiff Macklem": "Canada",
    # --- Non-state actors ---
    "Hamas": "Hamas",
    "Hezbollah": "Hezbollah",
    "Houthis": "Houthis",
    "Houthi": "Houthis",
    # --- Companies ---
    "Saudi Aramco": "Saudi Aramco",
    "Aramco": "Saudi Aramco",
    "Chevron": "Chevron",
    "Shell": "Shell",
    "BP": "BP",
    "ExxonMobil": "ExxonMobil",
    "Exxon Mobil": "ExxonMobil",
    "Rosneft": "Rosneft",
    "Lukoil": "Lukoil",
    "TotalEnergies": "TotalEnergies",
    # Goldman Sachs mapped for traceability but excluded from CANONICAL_ENTITIES
    "Goldman Sachs": "Goldman Sachs",
    # --- Commodities / benchmarks ---
    "West Texas Intermediate": "WTI",
    "U.S. West Texas Intermediate crude": "WTI",
    "West Texas Intermediate (WTI)": "WTI",
    "WTI": "WTI",
    "Brent": "Brent",
    "Brent crude": "Brent",
    "Brent Crude": "Brent",
    # --- Market indices ---
    "S&P 500": "S&P 500",
}

CANONICAL_ENTITIES: list[str] = [
    # Countries
    "US",
    "Iran",
    "Russia",
    "China",
    "Israel",
    "India",
    "Saudi Arabia",
    "Ukraine",
    "Venezuela",
    "Canada",
    "Iraq",
    "Nigeria",
    "UAE",
    "Kazakhstan",
    "Qatar",
    "Oman",
    "Japan",
    "Kuwait",
    "Pakistan",
    "Libya",
    "Mexico",
    "Azerbaijan",
    "Yemen",
    "Lebanon",
    "Brazil",
    "South Korea",
    "Guyana",
    "UK",
    "Algeria",
    "Germany",
    "Australia",
    "Hungary",
    "Egypt",
    "Türkiye",
    # Geographic regions / strategic locations
    "Strait of Hormuz",
    "Middle East",
    "Gaza",
    "Red Sea",
    "Persian Gulf",
    "Gulf of Mexico",
    "Permian Basin",
    "Europe",
    "Asia",
    # Political actors and individuals
    "Trump",
    "Maduro",
    "Putin",
    "Powell",
    "Bessent",
    "Biden",
    # Organizations
    "OPEC+",
    "OPEC",
    "Fed",
    "EU",
    "EIA",
    "IEA",
    "API",
    "ECB",
    # Non-state actors
    "Hamas",
    "Hezbollah",
    "Houthis",
    # Companies
    "Saudi Aramco",
    "Chevron",
    "Shell",
    "BP",
    "ExxonMobil",
    "Rosneft",
    "Lukoil",
    "TotalEnergies",
    # Commodities / benchmarks / indices
    "WTI",
    "Brent",
    "S&P 500",
]


_CANONICAL_SET = set(CANONICAL_ENTITIES)


def canonicalize_entities(entities_json: str | None) -> list[str]:
    """Map raw LLM entity strings to canonical names.

    Takes the JSON array stored in llm_features.entities and returns a
    deduplicated list of canonical entity names present in CANONICAL_ENTITIES.
    Entities not in ENTITY_CANONICAL, or whose canonical name is not in
    CANONICAL_ENTITIES (e.g. Goldman Sachs), are silently dropped.
    """
    if not entities_json:
        return []
    try:
        raw = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for entity in raw:
        canonical = ENTITY_CANONICAL.get(str(entity).strip())
        if canonical and canonical in _CANONICAL_SET and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


# ── Extraction helpers ─────────────────────────────────────────────────────────


def _user_text(title: str, body: str | None) -> str:
    title = str(title).encode("utf-8", errors="ignore").decode("utf-8").strip()
    body_text = ""
    if isinstance(body, str):
        body_text = body.encode("utf-8", errors="ignore").decode("utf-8")[:1500].strip()
    return f"Title: {title}\n\nBody: {body_text}" if body_text else f"Title: {title}"


def _parse_tool_input(tool_input: dict) -> dict:
    result = {"usable": tool_input["usable"]}
    if result["usable"]:
        for field in _OPTIONAL_FIELDS:
            result[field] = tool_input.get(field, None)
    else:
        for field in _OPTIONAL_FIELDS:
            result[field] = None
    return result


def build_batch_request(article_id: int, title: str, body: str | None) -> dict:
    """Return one Batches API request dict for the given article."""
    return {
        "custom_id": str(article_id),
        "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 500,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [EXTRACTION_TOOL],
            "tool_choice": {"type": "tool", "name": "extract_article_features"},
            "messages": [{"role": "user", "content": _user_text(title, body)}],
        },
    }


def parse_batch_result(result) -> dict | None:
    """
    Parse one MessageBatchResult from client.messages.batches.results().
    Returns the feature dict (keyed by article_id) or None if the result errored.
    """
    if result.result.type != "succeeded":
        return None
    tool_input = None
    for block in result.result.message.content:
        if block.type == "tool_use":
            tool_input = block.input
            break
    if tool_input is None:
        return None
    features = _parse_tool_input(tool_input)
    features["article_id"] = int(result.custom_id)
    return features


def extract_features(title: str, body: str | None, client: anthropic.Anthropic) -> dict:
    """Single-article synchronous extraction. Used for calibration runs."""
    response = client.messages.create(**build_batch_request(0, title, body)["params"])

    tool_input = None
    for block in response.content:
        if block.type == "tool_use":
            tool_input = block.input
            break
    if tool_input is None:
        raise ValueError(f"No tool_use block in response for title: {title[:50]}")

    return _parse_tool_input(tool_input)
