import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client

# -------------------------------------------------------------
# Load secrets
# -------------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="GPT5-Trade Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 GPT5-Trade — Live Trading Dashboard")


# -------------------------------------------------------------
# Helper: fetch tables
# -------------------------------------------------------------
@st.cache_data(ttl=30)
def load_trades():
    resp = supabase.table("trades").select("*").order("ts", desc=True).execute()
    return pd.DataFrame(resp.data or [])

@st.cache_data(ttl=30)
def load_shadow():
    resp = supabase.table("ml_shadow_logs").select("*").order("ts", desc=True).execute()
    return pd.DataFrame(resp.data or [])


# -------------------------------------------------------------
# Auto-refresh toggle
# -------------------------------------------------------------
col1, col2 = st.columns([1,3])
with col1:
    refresh = st.checkbox("Auto-refresh (30s)", value=True)

if refresh:
    st.experimental_rerun()


# -------------------------------------------------------------
# Load data
# -------------------------------------------------------------
trades_df = load_trades()
shadow_df = load_shadow()

if trades_df.empty:
    st.warning("No trades found in database.")
else:
    # -------------------------------------------------------------
    # SECTION 1 — Summary Stats
    # -------------------------------------------------------------
    st.subheader("📊 Summary Statistics")

    trades_df["pnl"] = trades_df["pnl"].astype(float)

    total_pnl = trades_df["pnl"].sum()
    total_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    win_rate = wins / max(1, (wins + losses))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total PnL", f"${total_pnl:,.2f}")
    kpi2.metric("Total Trades", total_trades)
    kpi3.metric("Win Rate", f"{win_rate*100:.1f}%")
    kpi4.metric("Wins / Losses", f"{wins} / {losses}")

    st.divider()


    # -------------------------------------------------------------
    # SECTION 2 — PnL Over Time
    # -------------------------------------------------------------
    st.subheader("📈 PnL Over Time")

    df_time = trades_df.copy()
    df_time["ts"] = pd.to_datetime(df_time["ts"])
    df_time = df_time.sort_values("ts")
    df_time["cumulative_pnl"] = df_time["pnl"].cumsum()

    fig = px.line(df_time, x="ts", y="cumulative_pnl", title="Cumulative PnL")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()


    # -------------------------------------------------------------
    # SECTION 3 — Symbol-Level Performance
    # -------------------------------------------------------------
    st.subheader("🎯 Performance by Symbol")

    sym = trades_df.groupby("symbol").agg(
        trades=("symbol", "count"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum())
    ).reset_index()

    fig2 = px.bar(sym, x="symbol", y="pnl", color="pnl", title="PnL by Symbol", color_continuous_scale="Turbo")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(sym, use_container_width=True)

    st.divider()


    # -------------------------------------------------------------
    # SECTION 4 — Real Trades Table
    # -------------------------------------------------------------
    st.subheader("📘 Live Trade Log")

    show_cols = [
        "ts", "symbol", "side", "qty", "fill_price", "pnl",
        "pnl_pct", "reasoning", "confidence", "is_entry", "is_exit"
    ]

    st.dataframe(trades_df[show_cols], use_container_width=True, height=400)

    st.divider()


# -------------------------------------------------------------
# SECTION 5 — ML SHADOW MODE VISUALS
# -------------------------------------------------------------
st.subheader("🧠 ML Shadow-Mode Predictions")

if shadow_df.empty:
    st.info("No ML shadow logs found.")
else:

    shadow_df["ts"] = pd.to_datetime(shadow_df["ts"])

    # ----- Latest Predictions -----
    st.markdown("### Latest 20 ML Predictions")
    st.dataframe(shadow_df.head(20), height=300, use_container_width=True)

    # ----- Direction distribution -----
    st.markdown("### ML Direction Distribution")

    dir_counts = shadow_df["ml_direction"].value_counts()
    fig3 = px.pie(
        names=dir_counts.index,
        values=dir_counts.values,
        title="Shadow-Mode Predicted Actions"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ----- Win probability histogram -----
    st.markdown("### Win Probability Distribution")

    shadow_df["ml_win_prob"] = shadow_df["ml_win_prob"].astype(float)

    fig4 = px.histogram(
        shadow_df,
        x="ml_win_prob",
        nbins=30,
        title="ML Predicted Win Probability Histogram"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()


# -------------------------------------------------------------
# SECTION 6 — Real vs Shadow Comparison
# -------------------------------------------------------------
st.subheader("⚖️ Real vs Shadow Decisions")

if not trades_df.empty and not shadow_df.empty:

    # Most recent shadow logs per symbol
    latest_shadow = shadow_df.sort_values("ts").groupby("symbol").tail(1)

    comparison_rows = []
    for _, row in latest_shadow.iterrows():
        comparison_rows.append({
            "symbol": row["symbol"],
            "shadow_dir": row["ml_direction"],
            "shadow_win_prob": row["ml_win_prob"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True)

else:
    st.info("Need at least one trade + one shadow log to compare.")

st.write("— End of Dashboard —")
