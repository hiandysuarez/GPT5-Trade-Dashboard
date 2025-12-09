import os
from datetime import datetime, date, time as dtime
from typing import Optional

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh
import altair as alt


# ============================================================
# Auto-refresh every 60 seconds
# ============================================================
st_autorefresh(interval=60_000, key="refresh")


# ============================================================
# Supabase Client
# ============================================================

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        st.error("❌ Missing Supabase credentials.")
        st.stop()


sb = get_supabase_client()


# ============================================================
# Helpers
# ============================================================

def normalize_ts(df: pd.DataFrame):
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def normalize_exit_flag(val):
    if val is True:
        return True
    if val in [False, None]:
        return False
    return str(val).strip().lower() in ("true", "1", "t", "yes", "y")


# ============================================================
# Fetch functions
# ============================================================

@st.cache_data(ttl=10)
def fetch_trades(symbol: Optional[str], day: Optional[date]):
    try:
        q = sb.table("trades").select("*")

        if symbol:
            q = q.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, dtime.min)
            end = datetime.combine(day, dtime.max)
            q = q.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        df = pd.DataFrame(q.order("ts", desc=False).execute().data or [])
        return normalize_ts(df)

    except Exception as e:
        st.error(f"❌ Error fetching trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def fetch_shadow(symbol: Optional[str], day: Optional[date]):
    try:
        q = sb.table("ml_shadow_logs").select("*")

        if symbol:
            q = q.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, dtime.min)
            end = datetime.combine(day, dtime.max)
            q = q.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        df = pd.DataFrame(q.order("ts", desc=False).execute().data or [])
        return normalize_ts(df)

    except Exception as e:
        st.error(f"❌ Error fetching shadow logs: {e}")
        return pd.DataFrame()


# ============================================================
# UI Setup
# ============================================================

st.set_page_config(page_title="GPT5-Trade Dashboard", layout="wide")

st.title("📊 GPT5-Trade Monitoring Dashboard")
st.caption("Realtime view for entries, exits, P&L, and ML shadow predictions")

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading Date", today)

df_for_symbols = fetch_trades(None, selected_date)
symbols = sorted(df_for_symbols["symbol"].dropna().unique().tolist()) if "symbol" in df_for_symbols else []

symbol = st.sidebar.selectbox("Symbol", ["(All)"] + symbols)
symbol = None if symbol == "(All)" else symbol

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.experimental_rerun()


# ============================================================
# Load Data
# ============================================================

df_trades = fetch_trades(symbol, selected_date)
df_shadow = fetch_shadow(symbol, selected_date)


# ============================================================
# Daily Summary
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades for this date.")
else:
    df_trades["is_exit_norm"] = df_trades["is_exit"].apply(normalize_exit_flag)
    exits = df_trades[df_trades["is_exit_norm"] == True].copy()

    pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else "pnl"
    total_pnl = exits[pnl_col].fillna(0).sum()

    wins = exits["win"].fillna(False).sum() if "win" in exits.columns else 0
    total_exits = len(exits)
    win_rate = (wins / total_exits * 100) if total_exits else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exit Trades", total_exits)
    c2.metric("Total P&L", f"{total_pnl:,.2f}")
    c3.metric("Wins", int(wins))
    c4.metric("Win Rate", f"{win_rate:.1f}%")


    # -------- Static P&L by Symbol Chart --------
    st.markdown("### 💰 P&L by Symbol")

    if not exits.empty:
        pnl_by_sym = (
            exits.groupby("symbol")[pnl_col]
            .sum()
            .reset_index()
            .rename(columns={pnl_col: "total_pnl"})
        )

        chart = (
            alt.Chart(pnl_by_sym)
            .mark_bar(size=50)
            .encode(
                x=alt.X("symbol:N", title="Symbol"),
                y=alt.Y("total_pnl:Q", title="Total P&L", scale=alt.Scale(zero=False)),
                color=alt.condition("datum.total_pnl > 0", alt.value("#00cc66"), alt.value("#cc0000"))
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)


# ============================================================
# Latest 10 Exit Trades (Global)
# ============================================================

st.subheader("📜 Latest 10 Exit Trades (Global, P&L Colored)")

df_all = fetch_trades(None, None)
df_all["is_exit_norm"] = df_all["is_exit"].apply(normalize_exit_flag)

# pick PnL column
pnl_col_global = "realized_pnl" if "realized_pnl" in df_all.columns else "pnl"

df_exits = df_all[(df_all["is_exit_norm"] == True) & df_all[pnl_col_global].notna()].copy()

if df_exits.empty:
    st.info("No exit trades available.")
else:
    df_exits = df_exits.sort_values("ts", ascending=False).head(10)

    # ---- Color only the PnL cells ----
    def highlight_pnl(val):
        try:
            v = float(val)
            if v > 0:
                return "background-color: rgba(0,255,0,0.25);"
            if v < 0:
                return "background-color: rgba(255,0,0,0.25);"
        except:
            pass
        return ""

    # Color both pnl + realized_pnl columns when they exist
    pnl_columns = [col for col in ["pnl", "realized_pnl"] if col in df_exits.columns]

    styled = df_exits.style.applymap(
        highlight_pnl,
        subset=pnl_columns
    )


    # Force single-line, horizontal scroll
    st.markdown(
        """
        <style>
        .stDataFrame td {
            white-space: nowrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(styled, height=350, hide_index=True)


# ============================================================
# Shadow Logs
# ============================================================

st.subheader("🧪 ML Shadow-Mode Logs")

if df_shadow.empty:
    st.info("No shadow logs.")
else:
    st.metric("Shadow log entries", len(df_shadow))

    if "ml_direction" in df_shadow.columns:
        counts = df_shadow["ml_direction"].fillna("UNKNOWN").value_counts()
        st.bar_chart(counts)

    cols = ["ts", "symbol", "ml_direction", "ml_win_prob", "bot_action"]
    cols = [c for c in cols if c in df_shadow.columns]

    st.dataframe(df_shadow.sort_values("ts", ascending=False)[cols], hide_index=True)


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("Powered by Supabase • GPT5-Trade • ML Shadow Mode • Streamlit Dashboard")
