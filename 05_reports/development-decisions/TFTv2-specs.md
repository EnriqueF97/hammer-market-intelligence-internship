Build the TFT v2 training notebook for the WTI news-driven liquidity thesis.

File: 02_notebooks/13_tft_v2_training.ipynb

CONTEXT

- This is Task 5 of the TFT v2 prep work. Tasks 1-4 are complete. Read project_logbook.md and thesis_decisions_log.md for full context.
- Constants live in 03_src/tft/config.py (already created). Use them, don't redefine.
- The notebook runs three times by toggling ABLATION_VARIANT at the top to 'v2.0', 'v2.1', or 'v2.2'. Each variant trains a separately-checkpointed model.
- Target compute: Google Colab Pro (A100/L4/V100). The DB file (wti_thesis.db) is on Google Drive. Local execution should also work as a fallback, with smaller batch.

NOTEBOOK STRUCTURE (17 cells, in order)

Cell 1 - Markdown header explaining the notebook's purpose, the three ablation variants, and how to run it.

Cell 2 - Imports. Standard ML stack: pandas, numpy, sqlite3, pytorch, pytorch-lightning, pytorch-forecasting, matplotlib for any quick visualizations. Also import the constants from src.tft.config.

Cell 3 - ABLATION_VARIANT toggle (set to 'v2.0', 'v2.1', or 'v2.2') and config verification. Call verify_against_db() if the helper exists in config.py. Define variant-dependent feature flags here (a dict mapping variant -> feature configuration) so the rest of the notebook reads cleanly.

Cell 4 - Load the modeling data. SQL JOIN of market_context, liquidity, llm_features (filtered by usable=1), and article_entities. Aggregate to one row per hour:

- Continuous LLM features: mean across articles in that hour (sentiment_score, supply_impact, demand_impact, risk_premium, magnitude, certainty)
- Binary entity flags: max across articles (top 71 canonical entities from article_entities)
- Categorical features: take the event_type[0] and time_horizon from the article with the highest magnitude in that hour. For event_type, treat the JSON array stored in llm_features.event_type; parse and take the first element.
- n_articles: count of articles aligned to that hour
- Hours with no articles: fill continuous features with 0.0, binary flags with 0, categoricals with a special 'no_news' label, n_articles with 0
- Market context (log_volume, amihud, price_range, log_return, dxy, vix): from market_context directly, no aggregation needed
- Calendar features (hour, day_of_week, month, is_us_session, is_wednesday): derived from datetime_hour
  Add time_idx column = row index (0 to TOTAL_HOURS-1) and asset column = 'WTI' as required by pytorch-forecasting.

Cell 5 - Apply train/val/test split using config constants. Compute slice indices for the regime-change analysis (pre-war test = TEST_START to WAR_ONSET_IDX, war test = WAR_ONSET_IDX to TEST_END). No data is dropped, the split is via time_idx ranges passed to TimeSeriesDataSet.

Cell 6 - Construct three TimeSeriesDataSet objects (train, val, test) with variant-specific feature lists. The variant configuration:

For v2.0: - target = ['log_volume'] - max_prediction_length = 1 - time_varying_known_reals = ['hour', 'day_of_week', 'month', 'is_us_session', 'is_wednesday'] - time_varying_unknown_reals = ['log_volume', 'price_range', 'log_return', 'amihud', 'dxy', 'vix', 'sentiment_score', 'supply_impact', 'demand_impact', 'risk_premium', 'magnitude', 'certainty', 'n_articles'] - time_varying_unknown_categoricals = [] - event_type and time_horizon enter as INT-ENCODED continuous reals (manually convert categorical labels to ints, including 'no_news' = 0). Add 'event_type_int', 'time_horizon_int' to time_varying_unknown_reals. - No entity flags. - static_categoricals = ['asset']

For v2.1: - Same as v2.0 EXCEPT event_type and time_horizon are proper categoricals, moved to time_varying_known_categoricals (yes, known - they're properties of the news at that hour, which is observed by the time we predict the next hour). Remove the int-encoded versions. - Still no entity flags.

For v2.2: - Same as v2.1 PLUS: - target = ['log_volume', 'amihud', 'price_range'] - max_prediction_length = 28 (encoder remains 48) - Add 71 entity flag columns to time_varying_unknown_reals (the flags are computed at the article's publication hour, so they're 'unknown' in the technical pytorch-forecasting sense; the model uses them through the encoder).

All three variants share: - max_encoder_length = ENCODER_LENGTH (48) - time_idx = 'time_idx' - group_ids = ['asset'] - allow_missing_timesteps = False - add_relative_time_idx = True - add_target_scales = True - add_encoder_length = True - Use the locked time_idx ranges from Cell 5 for train/val/test boundaries.

Cell 7 - Instantiate TemporalFusionTransformer with v1-matched architecture: - hidden_size = 32 - attention_head_size = 4 - dropout = 0.1 - hidden_continuous_size = 16 - learning_rate = 1e-3 - loss = QuantileLoss() (default 7 quantiles) - reduce_on_plateau_patience = 3 - log_interval = 10 - log_val_interval = 1 - Use TemporalFusionTransformer.from_dataset(train_dataset, ...) so the model is configured to the variant's feature set automatically.

Cell 8 - Set up Trainer: - max*epochs = 50 (early stopping will halt earlier) - gradient_clip_val = 0.1 - accelerator = 'gpu' if available else 'cpu' - callbacks:
* EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4)
* ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
dirpath='01_data/models/',
filename=f'tft*{ABLATION_VARIANT}-{{epoch}}-{{val_loss:.4f}}') - logger = CSVLogger to '04_outputs/tft_v2/{variant}/logs/' for the training curves - deterministic = True with a fixed seed (seed=42 via pytorch_lightning.seed_everything)

Cell 9 - trainer.fit(model, train_dataloader, val_dataloader). This is the long-running cell. Expected duration on Colab Pro A100: 30 min for v2.0/v2.1, 1.5-2 hr for v2.2 (due to multi-target multi-horizon expansion).

Cell 10 - Save final best checkpoint path to a variable. Print confirmation.

Cell 11 - Load the best checkpoint for evaluation. Use TemporalFusionTransformer.load_from_checkpoint(best_path).

Cell 12 - Generate predictions on val and test sets: - val_predictions = model.predict(val_dataloader, return_index=True, return_x=True, return_y=True) - test_predictions = same on test_dataloader - For v2.2, predictions come back as a dict keyed by target name; handle that case.

Cell 13 - Compute metrics: - For each (target, horizon, slice) compute MAE, RMSE, QuantileLoss (median quantile). - Slices: val, test_full, test_prewar, test_war (use WAR_ONSET_IDX to split test). - For v2.0 and v2.1: target = log_volume, horizons = [1]. So metrics are 1 target _ 1 horizon _ 4 slices = 4 rows. - For v2.2: 3 targets _ 5 horizons _ 4 slices = 60 rows. - Persist as a DataFrame and save to '04_outputs/tft_v2/{variant}/metrics.csv'. - Print a summary table to the notebook.

Cell 14 - Extract feature importance from the VSN. pytorch-forecasting exposes this via model.predict(..., mode='raw') or similar; consult library docs. Aggregate to per-feature mean importance over the val set. Save to '04_outputs/tft_v2/{variant}/feature_importance.json' as a dict {feature_name: importance_weight}.

Cell 15 - Extract attention weights: - Mean attention by lag, averaged over the val set: a 48-element array. - Mean attention by lag, split by sentiment direction at the prediction hour: three 48-element arrays (bearish/bullish/neutral). - Save to '04_outputs/tft_v2/{variant}/attention.npz' with named arrays.

Cell 16 - Save analysis artifacts: - Pickle predictions dict to '04_outputs/tft_v2/{variant}/predictions.pkl' - Save the best checkpoint path and key hyperparameters to '04_outputs/tft_v2/{variant}/run_metadata.json' for reproducibility.

Cell 17 - Summary print: - Best epoch - Best val_loss - Test loss (full, prewar, war) - Top 5 features by importance - Peak attention lag - Print clear "v2.X RUN COMPLETE" footer.

GENERAL RULES

- Use config constants from src.tft.config; do not redefine TRAIN_END, etc.
- Handle Colab path differences: detect if running on Colab and prepend /content/drive/MyDrive/ to relative paths accordingly. Use a PROJECT_ROOT variable at the top of cell 2.
- Use pytorch_lightning.seed_everything(42) for reproducibility.
- All filesystem writes go through pathlib.Path with parents=True, exist_ok=True.
- Keep cells focused: one logical step per cell, brief markdown headers above each.
- Where pytorch-forecasting has API ambiguity, prefer the more recent (post-1.0) API conventions.

ABLATION RUN SEQUENCE

1. Run v2.0 end-to-end. Verify metrics look sane (val_loss should be in the same order of magnitude as v1's 0.204).
2. Without restarting the kernel, change ABLATION_VARIANT to 'v2.1' and re-run from Cell 3 onward.
3. Same for v2.2.

After all three runs, the three sets of artifacts in 04_outputs/tft_v2/ are the substrate for §4.3.7 and §4.3.8 drafting.
