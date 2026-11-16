# Energy Dashboard v3
# ====================
# Based on v2. Adds a "Predict next hour" button in the Prediction area.
# Uses models/gbr_model.joblib and models/feature_order.json — no retraining.
# Each meter's prediction is built from its own revealed readings only.
#
# Run from the project root:
#   ..\vibe\Scripts\streamlit.exe run dashboard/app_v3.py --server.headless true

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

raw_url = os.environ["DATABASE_URL"]
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

JAN_START = datetime(2016, 1, 1, 0, 0)
JAN_END   = datetime(2016, 1, 31, 23, 0)

METER_COLORS = {
    "METER_01": "#4F86C6",
    "METER_02": "#F28C38",
    "METER_03": "#6BBF59",
}
TOTAL_COLOR = "#A855F7"

BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "gbr_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_order.json"

# Lag offsets required by the model (in hours)
LAG_HOURS = [1, 2, 3, 24, 168]
# Minimum rows needed per meter to satisfy all lags
MIN_ROWS_REQUIRED = 168

# ---------------------------------------------------------------------------
# Load model once at startup (cached as a resource)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model         = joblib.load(MODEL_PATH)
    feature_order = json.loads(FEATURES_PATH.read_text())
    return model, feature_order

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def fetch_january() -> pd.DataFrame:
    sql = """
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= %s AND timestamp <= %s
        ORDER  BY timestamp, meter_id
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (JAN_START, JAN_END))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fetch_next_hour(after: datetime) -> pd.DataFrame:
    sql = """
        WITH next_ts AS (
            SELECT MIN(timestamp) AS ts
            FROM   meter_readings_source
            WHERE  timestamp > %s
        )
        SELECT  r.timestamp, r.meter_id, r.building_id, r.electricity_kwh
        FROM    meter_readings_source r
        JOIN    next_ts n ON r.timestamp = n.ts
        ORDER   BY r.meter_id
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (after,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_next_hour(df_meter: pd.DataFrame, meter_id: str):
    """
    Build features for the hour immediately after the last revealed reading
    and return (predicted_kwh, target_timestamp) or raise ValueError with
    a human-readable reason if the history is insufficient.
    """
    model, feature_order = load_model()

    # Work on a time-sorted series of kWh values indexed by timestamp
    series = (
        df_meter[["timestamp", "electricity_kwh"]]
        .sort_values("timestamp")
        .set_index("timestamp")["electricity_kwh"]
    )

    n_rows      = len(series)
    last_ts     = series.index[-1]
    target_ts   = last_ts + timedelta(hours=1)

    # Check we have enough history for the deepest lag (168 h)
    if n_rows < MIN_ROWS_REQUIRED:
        missing = MIN_ROWS_REQUIRED - n_rows
        raise ValueError(
            f"{meter_id}: need {MIN_ROWS_REQUIRED} hourly readings, "
            f"have {n_rows} — {missing} more hour(s) required."
        )

    # Build lag features; the series is already sorted ascending
    kwh_values = series.values  # numpy array

    def get_lag(h):
        idx = n_rows - h   # h steps back from the end
        if idx < 0:
            raise ValueError(
                f"{meter_id}: lag_{h}h is out of range (only {n_rows} rows available)."
            )
        return float(kwh_values[idx])

    features = {
        "hour_of_day": float(target_ts.hour),
        "day_of_week": float(target_ts.dayofweek),
        "lag_1h":      get_lag(1),
        "lag_2h":      get_lag(2),
        "lag_3h":      get_lag(3),
        "lag_24h":     get_lag(24),
        "lag_168h":    get_lag(168),
    }

    X = np.array([[features[f] for f in feature_order]])
    prediction = float(model.predict(X)[0])
    return prediction, target_ts

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_state():
    df_jan = fetch_january()
    st.session_state["df"]        = df_jan
    st.session_state["last_ts"]   = df_jan["timestamp"].max()
    st.session_state["finished"]  = False
    st.session_state["next_tick"] = datetime.now(timezone.utc).timestamp() + 60


if "df" not in st.session_state:
    init_state()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Dashboard v3",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Dashboard v3")
st.caption("Data source: Neon PostgreSQL  •  Readings are **historical** (Jan–Feb 2016)")

# ---------------------------------------------------------------------------
# Restart button
# ---------------------------------------------------------------------------

col_status, col_btn = st.columns([4, 1])

with col_btn:
    if st.button("🔄 Restart simulation", use_container_width=True):
        init_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Live dashboard fragment  (reruns every 60 s)
# ---------------------------------------------------------------------------

@st.fragment(run_every=60)
def live_dashboard():
    if not st.session_state["finished"]:
        next_df = fetch_next_hour(st.session_state["last_ts"].to_pydatetime())
        if next_df.empty:
            st.session_state["finished"] = True
        else:
            st.session_state["df"] = pd.concat(
                [st.session_state["df"], next_df], ignore_index=True
            )
            st.session_state["last_ts"] = next_df["timestamp"].max()
            st.session_state["next_tick"] = datetime.now(timezone.utc).timestamp() + 60

    df         = st.session_state["df"]
    current_ts = st.session_state["last_ts"]
    finished   = st.session_state["finished"]

    with col_status:
        if finished:
            st.success(
                f"Simulation complete — all data through "
                f"{current_ts.strftime('%d %b %Y %H:%M')} has been loaded.",
                icon="✅",
            )
        else:
            st.info(
                f"Showing data through **{current_ts.strftime('%d %b %Y %H:%M')}** "
                f"— fetching the next hour every minute.",
                icon="🕐",
            )

    # Per-meter charts
    st.subheader("Hourly Electricity Usage by Meter")
    meters = sorted(df["meter_id"].unique())
    cols = st.columns(len(meters))
    for col, meter_id in zip(cols, meters):
        df_meter = df[df["meter_id"] == meter_id].copy()
        color = METER_COLORS.get(meter_id, "#888888")
        fig = go.Figure(
            go.Bar(
                x=df_meter["timestamp"],
                y=df_meter["electricity_kwh"],
                marker_color=color,
                name=meter_id,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} kWh<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=meter_id, font=dict(size=14)),
            xaxis=dict(title="Time", showgrid=False),
            yaxis=dict(title="kWh", gridcolor="#2d2d2d"),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            margin=dict(l=40, r=10, t=40, b=40),
            showlegend=False,
            height=300,
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)

    # Total chart
    st.subheader("Total Electricity Usage – All Meters")
    df_total = (
        df.groupby("timestamp", as_index=False)["electricity_kwh"]
        .sum()
        .sort_values("timestamp")
    )
    fig_total = go.Figure(
        go.Scatter(
            x=df_total["timestamp"],
            y=df_total["electricity_kwh"],
            mode="lines",
            line=dict(color=TOTAL_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.15)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} kWh<extra></extra>",
        )
    )
    fig_total.update_layout(
        title=dict(text="Total – All Meters Combined", font=dict(size=14)),
        xaxis=dict(title="Time", showgrid=False),
        yaxis=dict(title="kWh", gridcolor="#2d2d2d"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=40, r=10, t=40, b=40),
        showlegend=False,
        height=320,
    )
    st.plotly_chart(fig_total, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Simulation state**\n\n"
        f"- Rows loaded: `{len(df)}`\n"
        f"- Latest timestamp: `{current_ts.strftime('%Y-%m-%d %H:%M')}`\n"
        f"- Status: `{'finished' if finished else 'running'}`"
    )


live_dashboard()

st.divider()

# ---------------------------------------------------------------------------
# Bottom row: Prediction  |  LLM Analysis
# ---------------------------------------------------------------------------

pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")

    if st.button("Predict next hour", key="predict_btn"):
        df_current = st.session_state["df"]
        meters     = sorted(df_current["meter_id"].unique())

        METER_ICONS = {
            "METER_01": "⚡",
            "METER_02": "🔋",
            "METER_03": "💡",
        }

        card_cols = st.columns(len(meters))
        for col, meter_id in zip(card_cols, meters):
            df_meter = df_current[df_current["meter_id"] == meter_id].copy()
            color    = METER_COLORS.get(meter_id, "#888888")
            icon     = METER_ICONS.get(meter_id, "📊")
            try:
                kwh, target_ts = predict_next_hour(df_meter, meter_id)
                col.markdown(
                    f"""
                    <div style="
                        background:#1a1a2e;
                        border-left: 4px solid {color};
                        border-radius:8px;
                        padding:16px 20px;
                        text-align:center;
                    ">
                        <div style="font-size:2rem">{icon}</div>
                        <div style="color:{color};font-weight:700;font-size:1rem;margin-top:6px">{meter_id}</div>
                        <div style="color:#aaa;font-size:0.78rem;margin-top:4px">{target_ts.strftime('%d %b %Y %H:%M')}</div>
                        <div style="color:#fff;font-size:1.6rem;font-weight:700;margin-top:8px">{kwh:.2f}</div>
                        <div style="color:#aaa;font-size:0.78rem">kWh</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except ValueError as e:
                col.warning(str(e), icon="⚠️")

with llm_col:
    st.subheader("🤖 LLM Analysis")
    st.info("AI-powered energy analysis will appear here in a future update.", icon="🤖")

# ---------------------------------------------------------------------------
# Countdown fragment
# ---------------------------------------------------------------------------

@st.fragment(run_every=1)
def sidebar_countdown():
    if st.session_state.get("finished", False):
        st.sidebar.caption("⏹ Simulation finished")
        return
    secs_left = max(0, int(
        st.session_state.get("next_tick", 0) - datetime.now(timezone.utc).timestamp()
    ))
    m, s = divmod(secs_left, 60)
    st.sidebar.caption(f"⏱ Next fetch in {m:01d}:{s:02d}")


sidebar_countdown()

# ---------------------------------------------------------------------------
# Sidebar run command
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Run command")
st.sidebar.code(
    ".."
    + "\\"
    + "vibe"
    + "\\"
    + "Scripts"
    + "\\"
    + "streamlit.exe run dashboard/app_v3.py --server.headless true",
    language="bash",
)
