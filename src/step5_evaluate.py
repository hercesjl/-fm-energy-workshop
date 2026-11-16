"""
Step 5 – Evaluate the model.
Compare GradientBoostingRegressor MAE and RMSE against a naive baseline
(predict = previous hour's consumption) on the same test rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Reproduce prepared data and split ────────────────────────────────────────
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "hour_of_day", "day_of_week",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"
]
TARGET_COL = "electricity_kwh"
TEST_HOURS = 14 * 24

train_df = df.iloc[:-TEST_HOURS].copy()
test_df  = df.iloc[-TEST_HOURS:].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

# ── Train model ───────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# ── Model predictions ─────────────────────────────────────────────────────────
y_pred_model = model.predict(X_test)

# ── Baseline: simply repeat the previous hour's reading ──────────────────────
y_pred_baseline = test_df["lag_1h"].values

# ── Metrics ───────────────────────────────────────────────────────────────────
model_mae  = mean_absolute_error(y_test, y_pred_model)
model_rmse = root_mean_squared_error(y_test, y_pred_model)

base_mae   = mean_absolute_error(y_test, y_pred_baseline)
base_rmse  = root_mean_squared_error(y_test, y_pred_baseline)

mae_improvement  = (base_mae  - model_mae)  / base_mae  * 100
rmse_improvement = (base_rmse - model_rmse) / base_rmse * 100

# ── Worst errors ──────────────────────────────────────────────────────────────
errors   = np.abs(y_test.values - y_pred_model)
worst_idx = np.argsort(errors)[-5:][::-1]

print("=" * 55)
print("EVALUATION ON TEST SET (16 Feb – 29 Feb 2016, 336 hours)")
print()
print(f"{'Metric':<8}  {'Baseline (lag_1h)':>18}  {'GBR Model':>12}  {'Improvement':>12}")
print("-" * 55)
print(f"{'MAE':<8}  {base_mae:>15.3f} kWh  {model_mae:>9.3f} kWh  {mae_improvement:>+9.1f}%")
print(f"{'RMSE':<8}  {base_rmse:>15.3f} kWh  {model_rmse:>9.3f} kWh  {rmse_improvement:>+9.1f}%")
print()
print("WHAT THESE MEAN")
print(f"  On average the model's predictions are off by {model_mae:.2f} kWh.")
print(f"  The baseline (just repeating last hour) is off by {base_mae:.2f} kWh.")
print(f"  The model reduces average error by {mae_improvement:.1f}%.")
print()
print("5 LARGEST MODEL ERRORS IN TEST SET")
print(f"  {'Timestamp':<22} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"  {'-'*50}")
for i in worst_idx:
    ts  = test_df["timestamp"].iloc[i]
    act = y_test.values[i]
    prd = y_pred_model[i]
    err = errors[i]
    print(f"  {str(ts):<22} {act:>8.2f} {prd:>10.2f} {err:>8.2f}")
