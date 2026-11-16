"""
Step 6 – Inspect example predictions and save the work.
Plot actual vs predicted over the two-week test window.
Save the fitted model to models/gbr_model.joblib and
the feature order to models/feature_order.json.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

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

# ── Train ─────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("FM Energy Workshop – Actual vs Predicted\n"
             "Two-week test window: 16 Feb – 29 Feb 2016",
             fontsize=13, fontweight="bold")

timestamps = test_df["timestamp"].values

# Top panel: actual vs predicted
ax1 = axes[0]
ax1.plot(timestamps, y_test.values, label="Actual",    color="#1f77b4", linewidth=1.2)
ax1.plot(timestamps, y_pred,        label="Predicted", color="#ff7f0e",
         linewidth=1.2, linestyle="--", alpha=0.85)
ax1.set_ylabel("Electricity (kWh)")
ax1.legend(loc="upper right")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.grid(axis="y", alpha=0.3)

# Bottom panel: absolute error
errors = np.abs(y_test.values - y_pred)
ax2 = axes[1]
ax2.fill_between(timestamps, errors, color="#d62728", alpha=0.5, label="Abs error")
mae = mean_absolute_error(y_test, y_pred)
ax2.axhline(mae, color="#d62728", linewidth=1, linestyle=":",
            label=f"MAE = {mae:.2f} kWh")
ax2.set_ylabel("Error (kWh)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plot_path = "models/actual_vs_predicted.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved → {plot_path}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = "models/gbr_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

# ── Save feature order ────────────────────────────────────────────────────────
feature_path = "models/feature_order.json"
with open(feature_path, "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"Feature order saved → {feature_path}")

# ── Confirm files and sizes ───────────────────────────────────────────────────
print()
print("=" * 55)
print("SAVED FILES")
for path in [model_path, feature_path, plot_path]:
    size_kb = os.path.getsize(path) / 1024
    print(f"  {path:<40}  {size_kb:6.1f} KB")

print()
print("FINAL METRICS (test set)")
rmse = root_mean_squared_error(y_test, y_pred)
print(f"  MAE : {mae:.3f} kWh")
print(f"  RMSE: {rmse:.3f} kWh")
