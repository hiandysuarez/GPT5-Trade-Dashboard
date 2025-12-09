import os
from datetime import datetime, date, time as dtime
from typing import Optional, List

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
# 1. Supabase Client Setup
# ============================================================

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error("❌ Supabase secrets missing.")
        st.stop()

    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Supabase client: {e}")
        return None


sb = get_supabase_client()
if sb is None:
    st.stop()


# ============================================================
# Utilities
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
# Page Config
# ============================================================

st.set_page_config(page_title="GPT5-Trade Dashboard", layout="wide")
st.title("📊 GPT5-Trade Monitoring Dashboard")
st.caption("Realtime view for entries, exits, P&L, and ML shadow predictions")


# ============================================================
# Fetch Helpers
# ============================================================

@st.cache_data(ttl=10)
def fetch_trades(symbol: Optional[str], day: Optional[date]):
    try:
        query = sb.table("trades").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, dtime.min)
            end = datetime.combine(day, dtime.max)
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        df = pd.DataFrame(query.order("ts", desc=False).execute().data or [])
        return normalize_ts(df)
    except Exception as e:
        st.error(f"❌ Error fetching trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def fetch_shadow(symbol: Optional[str], day: Optional[date]):
    try:
        query = sb.table("ml_shadow_logs").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, dtime.min)
            end = datetime.combine(day, dtime.max)
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        df = pd.DataFrame(query.order("ts", desc=False).execute().data or [])
        return normalize_ts(df)
    except Exception as e:
        st.error(f"❌ Error fetching shadow logs: {e}")
        return pd.DataFrame()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date", today)

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
# Daily P&L Summary
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades for this day.")
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


    # -------- Static P&L by Symbol (Altair) --------
    st.markdown("### 💰 P&L by Symbol")

    if exits.empty:
        st.info("No exits → No P&L chart.")
    else:
        pnl_by_sym = (
            exits.groupby("symbol")[pnl_col]
            .sum()
            .reset_index()
            .rename(columns={pnl_col: "total_pnl"})
        )

        chart = (
            alt.Chart(pnl_by_sym)
            .mark_bar(size=25)  # thinner bars
            .encode(
                x=alt.X("symbol:N", title="Symbol", sort=None),
                y=alt.Y("total_pnl:Q", title="Total P&L", scale=alt.Scale(zero=False)),
                color=alt.condition("datum.total_pnl > 0",
                                    alt.value("#00cc66"),  # green
                                    alt.value("#cc0000"))   # red
            )
            .properties(height=300)
        )

        st.altair_chart(chart, use_container_width=True)


# ============================================================
# Latest 10 Exit Trades (Global, Colored)
# ============================================================

st.subheader("📜 Latest 10 Exit Trades (Global, P&L Colored)")

# Fetch all trades ignoring date filter
df_all = fetch_trades(None, None)
df_all["is_exit_norm"] = df_all["is_exit"].apply(normalize_exit_flag)

pnl_col_global = "realized_pnl" if "realized_pnl" in df_all.columns else "pnl"

df_exits = df_all[(df_all["is_exit_norm"] == True) & df_all[pnl_col_global].notna()].copy()

if df_exits.empty:
    st.info("No exit trades.")
else:
    df_exits = df_exits.sort_values("ts", ascending=False).head(10)

    # ---- HTML colored table ----
    def color_row(row):
        v = row[pnl_col_global]
        try:
            v = float(v)
            if v > 0:
                color = "rgba(0,255,0,0.25)"
            elif v < 0:
                color = "rgba(255,0,0,0.25)"
            else:
                color = "transparent"
        except:
            color = "transparent"
        return f"background-color:{color};"

    html = "<table style='width:100%;border-collapse:collapse;'>"
    html += "<tr>" + "".join(f"<th style='padding:6px;border-bottom:1px solid #666;text-align:left;'>{col}</th>" for col in df_exits.columns) + "</tr>"

    for _, row in df_exits.iterrows():
        style = color_row(row)
        html += "<tr>"
        for col in df_exits.columns:
            html += f"<td style='padding:6px;{style}'>{row[col]}</td>"
        html += "</tr>"

    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Shadow Logs
# ============================================================

st.subheader("🧪 ML Shadow-Mode Logs")

if df_shadow.empty:
    st.info("No shadow logs.")
else:
    st.metric("Shadow logs", len(df_shadow))

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
