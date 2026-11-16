# Energy Dashboard v2
# ====================
# Simulates a live feed by replaying historical Neon data one hour at a time.
#
# Startup : loads all of January 2016 (744 hourly readings per meter).
# Replay  : every real minute, fetches the next recorded hour from Neon
#           (one reading per meter) and appends it to the charts.
# Stops   : when the last record in the dataset has been retrieved.
# Restart : the "Restart simulation" button resets to January data and
#           restarts the one-minute timer.
#
# Run from the project root:
#   ..\vibe\Scripts\streamlit.exe run dashboard/app_v2.py --server.headless true

import os
import re
from datetime import datetime, timezone

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
    """Return the three readings for the next recorded hour after *after*."""
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
# Page setup  (must come before any other st calls)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Dashboard v2")
st.caption("Data source: Neon PostgreSQL  •  Readings are **historical** (Jan–Feb 2016)")

# ---------------------------------------------------------------------------
# Restart button  (outside the fragment so it triggers a full rerun)
# ---------------------------------------------------------------------------

col_status, col_btn = st.columns([4, 1])

with col_btn:
    if st.button("🔄 Restart simulation", use_container_width=True):
        init_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Auto-updating fragment
# run_every=60 makes Streamlit rerun only this fragment every 60 seconds,
# without reloading the whole page or depending on the browser.
# ---------------------------------------------------------------------------

@st.fragment(run_every=60)
def live_dashboard():
    # --- Fetch next hour if the simulation is still running -----------------
    if not st.session_state["finished"]:
        next_df = fetch_next_hour(st.session_state["last_ts"].to_pydatetime())
        if next_df.empty:
            st.session_state["finished"] = True
        else:
            st.session_state["df"] = pd.concat(
                [st.session_state["df"], next_df], ignore_index=True
            )
            st.session_state["last_ts"] = next_df["timestamp"].max()
            # Record when the next fetch is due
            st.session_state["next_tick"] = datetime.now(timezone.utc).timestamp() + 60

    # --- Read current state -------------------------------------------------
    df         = st.session_state["df"]
    current_ts = st.session_state["last_ts"]
    finished   = st.session_state["finished"]

    # --- Status message -----------------------------------------------------
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

    # --- Per-meter charts ---------------------------------------------------
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

    # --- Total chart --------------------------------------------------------
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

    # --- Sidebar state ------------------------------------------------------
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
# Placeholders  (static — outside the fragment)
# ---------------------------------------------------------------------------

pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")
    st.info("Energy usage prediction will appear here in a future update.", icon="🔮")

with llm_col:
    st.subheader("🤖 LLM Analysis")
    st.info("AI-powered energy analysis will appear here in a future update.", icon="🤖")

# ---------------------------------------------------------------------------
# Countdown fragment — reruns every second, only updates the sidebar ticker
# ---------------------------------------------------------------------------

@st.fragment(run_every=1)
def sidebar_countdown():
    if st.session_state.get("finished", False):
        st.sidebar.caption("⏹ Simulation finished")
        return
    secs_left = max(0, int(st.session_state.get("next_tick", 0) - datetime.now(timezone.utc).timestamp()))
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
    + "streamlit.exe run dashboard/app_v2.py --server.headless true",
    language="bash",
)
