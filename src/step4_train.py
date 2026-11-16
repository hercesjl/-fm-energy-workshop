"""
Step 4 – Train a GradientBoostingRegressor on the training set.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# ── Reproduce prepared data and split (same logic as Steps 2 & 3) ─────────────
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
TEST_HOURS = 14 * 24  # 336

train_df = df.iloc[:-TEST_HOURS].copy()
test_df  = df.iloc[-TEST_HOURS:].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

# ── Train ─────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# ── Quick sanity check: R² on train and test ──────────────────────────────────
train_r2 = model.score(X_train, y_train)
test_r2  = model.score(X_test,  y_test)

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)

print("=" * 55)
print("MODEL TRAINED")
print(f"  Algorithm   : GradientBoostingRegressor")
print(f"  random_state: 42")
print(f"  Training rows used: {len(X_train)}")

print("\nR² SCORES  (1.0 = perfect, 0.0 = no better than the mean)")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test  R²: {test_r2:.4f}")

print("\nFEATURE IMPORTANCES (most → least influential)")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<14}  {imp:.4f}  {bar}")
