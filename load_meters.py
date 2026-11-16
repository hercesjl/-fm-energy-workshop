import csv
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

raw_url = os.environ["DATABASE_URL"]
# Strip channel_binding=require as instructed
conn_str = re.sub(r'[&?]channel_binding=require', '', raw_url)

# Derive HTTP SQL endpoint from the connection string
# postgresql://user:pass@host/dbname?... -> https://host/sql
match = re.match(r'postgresql://[^@]+@([^/]+)/.*', conn_str)
host = match.group(1)
sql_url = f"https://{host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

def run_sql(query, params=None):
    body = {"query": query}
    if params:
        body["params"] = params
    resp = requests.post(sql_url, json=body, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

# 1. Create table
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    timestamp       TIMESTAMP        NOT NULL,
    meter_id        TEXT             NOT NULL,
    building_id     TEXT             NOT NULL,
    electricity_kwh DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (meter_id, timestamp)
)
""")
print("Table ready.")

# 2. Load CSV and upload rows
csv_path = os.path.join(os.path.dirname(__file__), "data", "three_meters.csv")
rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append((
            row["timestamp"],
            row["meter_id"],
            row["building_id"],
            float(row["electricity_kwh"]),
        ))

print(f"Rows to insert: {len(rows)}")

# Insert in batches of 500
BATCH = 500
inserted = 0
for i in range(0, len(rows), BATCH):
    batch = rows[i:i+BATCH]
    # Build a multi-row VALUES clause with positional params
    placeholders = []
    params = []
    for j, r in enumerate(batch):
        base = j * 4
        placeholders.append(f"(${base+1}, ${base+2}, ${base+3}, ${base+4})")
        params.extend(r)
    sql = (
        "INSERT INTO meter_readings_source (timestamp, meter_id, building_id, electricity_kwh) "
        "VALUES " + ", ".join(placeholders) +
        " ON CONFLICT (meter_id, timestamp) DO NOTHING"
    )
    run_sql(sql, params)
    inserted += len(batch)
    print(f"  Inserted batch up to row {inserted}")

print("Upload complete.")

# 3. Summary per meter
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)                   AS row_count,
    MIN(timestamp)             AS earliest,
    MAX(timestamp)             AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

print("\n{:<12} {:>9}  {:<20}  {:<20}".format("meter_id", "row_count", "earliest", "latest"))
print("-" * 68)
for row in result["rows"]:
    print("{:<12} {:>9}  {:<20}  {:<20}".format(
        row["meter_id"], row["row_count"], str(row["earliest"]), str(row["latest"])
    ))
