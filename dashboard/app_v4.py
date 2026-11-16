# Energy Dashboard v4
# ====================
# Based on v3. Replaces the LLM Analysis placeholder with a real OpenAI
# integration for METER_02 energy usage summary.
#
# What's new vs v3:
#   - LLM Analysis panel has a fixed (read-only) prompt and a Send button.
#   - On Send: takes the latest 48 hours of METER_02 readings currently in
#     session state, computes total / average / min / max kWh, builds a JSON
#     context, and sends one request to OpenAI.
#   - The response is cached in session_state — refreshes and the 60-second
#     replay do NOT trigger new API calls.
#   - Displays: response text, "Based on readings up to <ts>", a request
#     counter for this session, and a collapsed expander showing the JSON.
#   - Reads OPENAI_API_KEY and OPENAI_MODEL from .env; never displays the key.
#
# Run from the project root:
#   ..\vibe\Scripts\streamlit.exe run dashboard/app_v4.py --server.headless true

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
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

raw_url = os.environ["DATABASE_URL"]
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.environ["OPENAI_MODEL"]

JAN_START = datetime(2016, 1, 1, 0, 0)
JAN_END   = datetime(2016, 1, 31, 23, 0)

METER_COLORS = {
    "METER_01": "#4F86C6",
    "METER_02": "#F28C38",
    "METER_03": "#6BBF59",
}
TOTAL_COLOR = "#A855F7"

BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_PATH    = BASE_DIR / "models" / "gbr_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_order.json"

LAG_HOURS        = [1, 2, 3, 24, 168]
MIN_ROWS_REQUIRED = 168

FIXED_PROMPT = "Give me a summary of the energy usage for METER_02 for the past 2 days"

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
    model, feature_order = load_model()

    series = (
        df_meter[["timestamp", "electricity_kwh"]]
        .sort_values("timestamp")
        .set_index("timestamp")["electricity_kwh"]
    )

    n_rows    = len(series)
    last_ts   = series.index[-1]
    target_ts = last_ts + timedelta(hours=1)

    if n_rows < MIN_ROWS_REQUIRED:
        missing = MIN_ROWS_REQUIRED - n_rows
        raise ValueError(
            f"{meter_id}: need {MIN_ROWS_REQUIRED} hourly readings, "
            f"have {n_rows} — {missing} more hour(s) required."
        )

    kwh_values = series.values

    def get_lag(h):
        idx = n_rows - h
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
# LLM helper — builds the data context and calls OpenAI
# ---------------------------------------------------------------------------

def build_meter02_context(df: pd.DataFrame) -> tuple[dict, str]:
    """
    Takes the latest 48 hours of METER_02 readings from df.
    Returns (context_dict, latest_timestamp_str).
    """
    df_m2 = (
        df[df["meter_id"] == "METER_02"]
        .sort_values("timestamp")
        .copy()
    )

    # Latest 48 rows (each row = 1 hour)
    df_48 = df_m2.tail(48).reset_index(drop=True)

    if df_48.empty:
        raise ValueError("No METER_02 readings available yet.")

    kwh_values = df_48["electricity_kwh"].tolist()
    total_kwh  = round(sum(kwh_values), 4)
    avg_kwh    = round(total_kwh / len(kwh_values), 4)
    min_kwh    = round(min(kwh_values), 4)
    max_kwh    = round(max(kwh_values), 4)

    start_ts = df_48["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_ts   = df_48["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M")

    hourly_readings = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "electricity_kwh": round(row["electricity_kwh"], 4),
        }
        for _, row in df_48.iterrows()
    ]

    context = {
        "meter": "METER_02",
        "time_range": {"from": start_ts, "to": end_ts},
        "statistics": {
            "total_kwh":   total_kwh,
            "average_kwh": avg_kwh,
            "min_kwh":     min_kwh,
            "max_kwh":     max_kwh,
            "num_readings": len(kwh_values),
        },
        "hourly_readings": hourly_readings,
    }

    return context, end_ts


def call_openai(prompt: str, context: dict) -> str:
    """Sends the fixed prompt + JSON context to OpenAI and returns the reply."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    user_message = (
        f"{prompt}\n\n"
        f"Here is the data context in JSON:\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an energy analyst assistant. "
                    "Analyse the provided meter data and give a clear, concise summary. "
                    "Use the statistics and hourly readings supplied; do not invent values."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_state():
    df_jan = fetch_january()
    st.session_state["df"]        = df_jan
    st.session_state["last_ts"]   = df_jan["timestamp"].max()
    st.session_state["finished"]  = False
    st.session_state["next_tick"] = datetime.now(timezone.utc).timestamp() + 60
    # LLM state — preserved across replays; cleared only on full restart
    if "llm_response" not in st.session_state:
        st.session_state["llm_response"]     = None
        st.session_state["llm_context_json"] = None
        st.session_state["llm_latest_ts"]    = None
        st.session_state["llm_request_count"] = 0


if "df" not in st.session_state:
    init_state()
else:
    # Ensure LLM keys exist even if the page reloads mid-session
    for key, default in [
        ("llm_response", None),
        ("llm_context_json", None),
        ("llm_latest_ts", None),
        ("llm_request_count", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Dashboard v4",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Dashboard v4")
st.caption("Data source: Neon PostgreSQL  •  Readings are **historical** (Jan–Feb 2016)")

# ---------------------------------------------------------------------------
# Restart button
# ---------------------------------------------------------------------------

col_status, col_btn = st.columns([4, 1])

with col_btn:
    if st.button("🔄 Restart simulation", use_container_width=True):
        # Reset everything including LLM state
        for k in ["llm_response", "llm_context_json", "llm_latest_ts", "llm_request_count"]:
            st.session_state.pop(k, None)
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

# ── Prediction (unchanged from v3) ─────────────────────────────────────────
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

# ── LLM Analysis ───────────────────────────────────────────────────────────
with llm_col:
    st.subheader("🤖 LLM Analysis")

    # Read-only prompt display
    st.text_area(
        label="Prompt",
        value=FIXED_PROMPT,
        height=80,
        disabled=True,
        label_visibility="collapsed",
        key="llm_prompt_display",
    )

    send_clicked = st.button("Send", key="llm_send_btn", type="primary")

    if send_clicked:
        df_now = st.session_state["df"]
        try:
            context, latest_ts_str = build_meter02_context(df_now)
            context_json = json.dumps(context, indent=2)

            with st.spinner("Asking the LLM…"):
                reply = call_openai(FIXED_PROMPT, context)

            # Persist in session state so reruns / 60-s replay don't re-call
            st.session_state["llm_response"]      = reply
            st.session_state["llm_context_json"]  = context_json
            st.session_state["llm_latest_ts"]     = latest_ts_str
            st.session_state["llm_request_count"] += 1

        except Exception as exc:
            st.error(f"LLM request failed: {exc}", icon="❌")

    # Display cached response (shown even without pressing Send again)
    if st.session_state["llm_response"]:
        latest_ts   = st.session_state["llm_latest_ts"]
        req_count   = st.session_state["llm_request_count"]
        context_json = st.session_state["llm_context_json"]

        st.caption(f"Based on readings up to **{latest_ts}**  •  LLM requests this session: **{req_count}**")
        st.markdown(st.session_state["llm_response"])

        with st.expander("Data sent to the LLM", expanded=False):
            st.code(context_json, language="json")

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
# Sidebar — LLM request counter + run command
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
req_count = st.session_state.get("llm_request_count", 0)
st.sidebar.markdown(f"**LLM requests this session:** `{req_count}`")

st.sidebar.markdown("## Run command")
st.sidebar.code(
    ".."
    + "\\"
    + "vibe"
    + "\\"
    + "Scripts"
    + "\\"
    + "streamlit.exe run dashboard/app_v4.py --server.headless true",
    language="bash",
)
