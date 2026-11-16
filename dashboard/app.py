# Energy Dashboard
# =================
# Reads historical meter readings from Neon PostgreSQL and displays:
#   - Three side-by-side hourly electricity charts (one per meter)
#   - A combined total-usage chart (sum of all three meters)
#   - Placeholder sections for Prediction and LLM Analysis
#
# Run from the project root:
#   ..\vibe\Scripts\streamlit.exe run dashboard/app.py --server.headless true

import os
import re

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
# Strip channel_binding=require (not supported by psycopg2)
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

METER_COLORS = {
    "METER_01": "#4F86C6",
    "METER_02": "#F28C38",
    "METER_03": "#6BBF59",
}
TOTAL_COLOR = "#A855F7"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner="Loading meter readings from Neon…")
def load_readings() -> pd.DataFrame:
    """Fetch all rows from meter_readings_source and return a tidy DataFrame."""
    sql = """
        SELECT timestamp, meter_id, building_id, electricity_kwh
        FROM meter_readings_source
        ORDER BY timestamp
    """
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def meter_chart(df_meter: pd.DataFrame, meter_id: str) -> go.Figure:
    """Bar chart of hourly electricity usage for a single meter."""
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
    return fig


def total_chart(df_total: pd.DataFrame) -> go.Figure:
    """Line chart of the summed electricity usage across all meters."""
    fig = go.Figure(
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
    fig.update_layout(
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
    return fig

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Dashboard")
st.caption("Data source: Neon PostgreSQL  •  All readings are **historical**")

# --- Load data --------------------------------------------------------------
try:
    df = load_readings()
except Exception as exc:
    st.error(f"Could not connect to the database: {exc}")
    st.stop()

meters = sorted(df["meter_id"].unique())

# --- Top row: one chart per meter side by side ------------------------------
st.subheader("Hourly Electricity Usage by Meter")

cols = st.columns(len(meters))
for col, meter_id in zip(cols, meters):
    df_meter = df[df["meter_id"] == meter_id].copy()
    with col:
        st.plotly_chart(
            meter_chart(df_meter, meter_id),
            use_container_width=True,
        )

# --- Total usage chart ------------------------------------------------------
st.subheader("Total Electricity Usage – All Meters")

df_total = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .sort_values("timestamp")
)
st.plotly_chart(total_chart(df_total), use_container_width=True)

st.divider()

# --- Placeholders -----------------------------------------------------------
pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")
    st.info("Energy usage prediction will appear here in a future update.", icon="🔮")

with llm_col:
    st.subheader("🤖 LLM Analysis")
    st.info("AI-powered energy analysis will appear here in a future update.", icon="🤖")

# ---------------------------------------------------------------------------
# Run command reminder (shown in sidebar)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "### How to run\n\n"
    "From the **project root**:\n\n"
    "```\n"
    "..\\\\vibe\\\\Scripts\\\\streamlit.exe run dashboard/app.py --server.headless true\n"
    "```\n"
)
