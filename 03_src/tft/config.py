# 03_src/tft/config.py
"""Configuration constants for TFT v2 training and evaluation.

These constants are shared across all ablation runs (v2.0, v2.1, v2.2) to
guarantee identical splits and evaluation slices.
"""

# Total hours and window
TOTAL_HOURS = 11232
ENCODER_LENGTH = 48
MAX_PREDICTION_LENGTH = 28

# Temporal split (row indices in market_context sorted by datetime_hour)
TRAIN_END = 7862  # 70% of total
VAL_START = 7910  # TRAIN_END + ENCODER_LENGTH (48h buffer)
VAL_END = 9547  # 85% of total
TEST_START = 9595  # VAL_END + ENCODER_LENGTH (48h buffer)
TEST_END = 11232

# Regime-change slice within test set
# 2026-03-01 23:00 UTC is the first market hour after the 28-Feb-2026 attack.
# Use this to split test metrics into pre-war and war subsets.
WAR_ONSET_IDX = 10056
WAR_ONSET_DATETIME = "2026-03-01 23:00:00+00:00"


def entity_to_column_name(canonical: str) -> str:
    """Canonical entity name → valid pandas column name with ent_ prefix.

    Uses unicode normalization so accented characters degrade gracefully
    (e.g. 'Türkiye' → 'ent_turkiye').  Inverse mapping is in COL_TO_ENTITY.
    """
    import re
    import unicodedata

    name = unicodedata.normalize("NFKD", canonical)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = name.replace(" ", "_")
    name = name.replace("+", "_plus")
    name = name.replace("&", "_and_")
    name = name.replace(".", "")
    name = name.replace("-", "_")
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return f"ent_{name}"


def _build_entity_maps() -> "tuple[dict, dict]":
    """Build canonical→column and column→canonical dicts from CANONICAL_ENTITIES."""
    import sys
    from pathlib import Path

    _src = Path(__file__).parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from nlp.llm_features import CANONICAL_ENTITIES  # type: ignore

    fwd = {e: entity_to_column_name(e) for e in CANONICAL_ENTITIES}
    rev = {v: k for k, v in fwd.items()}
    return fwd, rev


# Canonical entity name → column name  (e.g. 'Türkiye' → 'ent_turkiye')
ENTITY_COL_MAP: "dict[str, str]" = {}
# Column name → canonical entity name  (reverse of above, for reporting)
COL_TO_ENTITY: "dict[str, str]" = {}

try:
    ENTITY_COL_MAP, COL_TO_ENTITY = _build_entity_maps()
except Exception:
    pass  # maps stay empty if llm_features is unavailable (e.g. bare import)


def verify_against_db(db_path: str = "01_data/wti_thesis.db") -> None:
    """Sanity check: confirm TOTAL_HOURS matches the current DB state.

    Raises AssertionError if the market_context row count has drifted
    from the value the splits were computed against. Run this at the
    top of any training notebook.
    """
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        actual = conn.execute("SELECT COUNT(*) FROM market_context").fetchone()[0]
    assert actual == TOTAL_HOURS, (
        f"market_context has {actual} rows but TFT config expects {TOTAL_HOURS}. "
        f"Either the dataset changed (re-lock the split) or you're pointing at "
        f"the wrong DB."
    )
