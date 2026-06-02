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
