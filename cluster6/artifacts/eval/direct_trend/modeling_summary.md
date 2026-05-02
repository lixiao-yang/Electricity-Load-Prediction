# Cluster 6 MT_362 Direct Trend Forecasting Summary

## Data
- Target customer: `MT_362`.
- Timestamp field: DatetimeIndex named `timestamp`.
- Detected frequency: `h` with median step `0 days 01:00:00`.
- Model-train range: 2012-10-01 00:00:00 to 2014-06-30 23:00:00 (15,312 rows).
- Validation range: 2014-07-01 00:00:00 to 2014-09-30 23:00:00 (2,208 rows).
- Test range: 2014-10-01 00:00:00 to 2014-12-31 23:00:00 (2,208 rows).

## Why ETS/Holt-Winters Is Excluded
ETS/Holt-Winters was removed from model selection because its forecast for this customer behaved like a repeated daily seasonal curve and did not capture the desired level/trend behavior. It is not used as a candidate model in this notebook.

## Models Compared
- Naive baseline: repeats the most recent observed load.
- Direct multi-horizon LightGBM: supervised model with lag, rolling, calendar, horizon, and conservative trend-context features.
- Direct multi-horizon XGBoost: same conservative direct feature design with a boosted-tree implementation.
- Prophet: explicit trend plus daily, weekly, and yearly seasonal components.

## Feature Rationale
- Lag features capture autocorrelation.
- Rolling means/std/min/max capture recent level and volatility.
- Calendar features capture natural electricity usage cycles.
- `horizon`, rolling-level ratios, and recent slope features give the direct models trend context without directly extrapolating a long-horizon projected level.

## Rolling-Origin Validation Comparison
| model           | key_parameters                                                                                                                                                                           |   validation_mape | justification                                                                                                                                              |
|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| direct_xgboost  | {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_child_weight": 10, "tree_method": "hist"}                |           13.2539 | Direct multi-horizon tree model with lag, rolling, calendar, horizon, and conservative trend-context features; selected by rolling-origin validation MAPE. |
| direct_lightgbm | {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}                    |           13.6999 | Direct multi-horizon tree model with lag, rolling, calendar, horizon, and conservative trend-context features; selected by rolling-origin validation MAPE. |
| prophet         | {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 5.0, "seasonality_mode": "multiplicative", "daily_fourier_order": 10, "weekly_fourier_order": 5, "yearly_fourier_order": 5} |           20.0469 | Explicit trend plus daily, weekly, and yearly seasonality model; selected by rolling-origin validation MAPE from a small grid.                             |
| naive           | {"strategy": "last observed value repeated"}                                                                                                                                             |          114.501  | Simple benchmark that repeats the most recent observed load; selected/evaluated by rolling-origin validation MAPE.                                         |

## Final Test Evaluation
- Selected model: `direct_xgboost`.
- Selected parameters: `{"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_child_weight": 10, "tree_method": "hist"}`.
- Selection rule: lowest rolling-origin validation MAPE only.
- Rolling-origin validation MAPE: `13.2539%`.
- Single split validation MAPE for plotting/reference: `14.3641%`.
- Final test legacy MAPE: `15.6502%`.
- Final test MAPE_0_100: `15.6502%`.
- Final test EPSILON_MAPE_PCT: `15.6502%`.
- Final test WMAPE_0_100: `15.4142%`.

## Limitations and Future Improvements
- Direct multi-horizon training is more expensive than recursive forecasting, but it avoids feeding predictions back into lag features.
- Tree models still cannot know future structural changes without exogenous signals; weather, holidays, or operational covariates would likely improve trend/level-shift prediction. The aggressive projected-level extrapolation feature was removed to avoid forcing an overly sharp October drop.
- Prophet is useful for explicit trend modeling, but may be less accurate than boosted trees on high-frequency electricity data.
- All model and parameter choices use rolling-origin validation MAPE only; the test set is used once for final evaluation.
