"""
Step 2 – Prepare the data.
Build features: hour_of_day, day_of_week, lag_1h, lag_2h, lag_3h, lag_24h, lag_168h.
Target: electricity_kwh at hour t.
Drop the first 168 rows that lack full history.
"""

import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ── Calendar features ─────────────────────────────────────────────────────────
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week   # 0=Monday … 6=Sunday

# ── Lag features (past readings only – no future leakage) ─────────────────────
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

# ── Drop rows without full history ────────────────────────────────────────────
df = df.dropna().reset_index(drop=True)

# ── Define feature columns and target ────────────────────────────────────────
FEATURE_COLS = [
    "hour_of_day", "day_of_week",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"
]
TARGET_COL = "electricity_kwh"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ── Print checks ─────────────────────────────────────────────────────────────
print("=" * 55)
print("SHAPE AFTER FEATURE ENGINEERING")
print(f"  Rows: {len(df)}   Features: {len(FEATURE_COLS)}")

print(f"\nDATE RANGE AFTER DROP")
print(f"  From : {df['timestamp'].min()}")
print(f"  To   : {df['timestamp'].max()}")

print(f"\nFEATURE COLUMNS")
for col in FEATURE_COLS:
    print(f"  {col}")
print(f"  → target: {TARGET_COL}")

print("\nFIRST 2 ROWS OF FEATURE TABLE")
print(df[["timestamp"] + FEATURE_COLS + [TARGET_COL]].head(2).to_string(index=False))

print("\nMISSING VALUES IN FEATURES")
print(X.isnull().sum().to_string())

print("\nFEATURE STATISTICS")
print(X.describe().to_string())
