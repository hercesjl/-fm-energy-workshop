"""
Step 1 – Understand the data.
Load data/one_meter.csv and print a brief description.
"""

import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

print("=" * 55)
print("SHAPE")
print(f"  Rows: {len(df)}   Columns: {df.columns.tolist()}")

print("\nFIRST 3 ROWS")
print(df.head(3).to_string(index=False))

print("\nLAST 3 ROWS")
print(df.tail(3).to_string(index=False))

print("\nDATE RANGE")
print(f"  From : {df['timestamp'].min()}")
print(f"  To   : {df['timestamp'].max()}")
span = df['timestamp'].max() - df['timestamp'].min()
print(f"  Span : {span.days} days")

print("\nCOLUMN TYPES")
print(df.dtypes.to_string())

print("\nMISSING VALUES")
print(df.isnull().sum().to_string())

print("\nTIMESTAMP GAPS (missing hours)")
expected = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="h"
)
missing_ts = expected.difference(df["timestamp"])
if len(missing_ts) == 0:
    print("  None – every hour is present.")
else:
    print(f"  {len(missing_ts)} missing hours found:")
    print(missing_ts)

print("\nDUPLICATE TIMESTAMPS")
dupes = df["timestamp"].duplicated().sum()
print(f"  {dupes} duplicate(s)")

print("\nELECTRICITY_KWH STATISTICS")
print(df["electricity_kwh"].describe().to_string())
