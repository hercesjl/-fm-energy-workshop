"""
Step 3 – Split into train and test sets.
Hold back the last two weeks (336 hours) as the test set.
Train on everything before that.
"""

import pandas as pd

# ── Reproduce the prepared dataset from Step 2 ───────────────────────────────
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

# ── Split: last 2 weeks = 14 days × 24 hours = 336 rows ──────────────────────
TEST_HOURS = 14 * 24  # 336

train_df = df.iloc[:-TEST_HOURS].copy()
test_df  = df.iloc[-TEST_HOURS:].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

# ── Print checks ──────────────────────────────────────────────────────────────
print("=" * 55)
print("SPLIT SUMMARY")
print(f"  Total usable rows : {len(df)}")
print(f"  Train rows        : {len(train_df)}")
print(f"  Test  rows        : {len(test_df)}")
print(f"  Test share        : {len(test_df)/len(df)*100:.1f}%")

print("\nTRAIN DATE RANGE")
print(f"  From : {train_df['timestamp'].min()}")
print(f"  To   : {train_df['timestamp'].max()}")

print("\nTEST DATE RANGE")
print(f"  From : {test_df['timestamp'].min()}")
print(f"  To   : {test_df['timestamp'].max()}")

print("\nNO OVERLAP CHECK")
train_end  = train_df["timestamp"].max()
test_start = test_df["timestamp"].min()
gap_hours  = (test_start - train_end).total_seconds() / 3600
print(f"  Last train hour  : {train_end}")
print(f"  First test hour  : {test_start}")
print(f"  Gap between them : {gap_hours:.0f} hour(s)  ✓" if gap_hours == 1 else f"  Gap: {gap_hours} hours — check this!")
