import os
from datetime import datetime, date, time as dtime
from typing import Optional, List

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh


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
        st.error("❌ Supabase secrets missing. Add SUPABASE_URL + SUPABASE_KEY.")
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
# Utility: Normalize timestamps
# ============================================================

def normalize_ts(df: pd.DataFrame):
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


# ============================================================
# 2. Page Config
# ============================================================

st.set_page_config(
    page_title="GPT5-Trade Dashboard",
    layout="wide",
)

st.title("📊 GPT5-Trade Monitoring Dashboard")
st.caption("Realtime view for entries, exits, P&L, and ML shadow predictions")


# ============================================================
# 3. Fetch Helpers
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

        resp = query.order("ts", desc=False).execute()
        df = pd.DataFrame(resp.data or [])
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

        resp = query.order("ts", desc=False).execute()
        df = pd.DataFrame(resp.data or [])
        return normalize_ts(df)

    except Exception as e:
        st.error(f"❌ Error fetching shadow logs: {e}")
        return pd.DataFrame()


# ============================================================
# 4. Sidebar
# ============================================================

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date", today)

df_all = fetch_trades(None, selected_date)

symbols = sorted(df_all["symbol"].dropna().unique().tolist()) if "symbol" in df_all else []
symbol = st.sidebar.selectbox("Symbol", ["(All)"] + symbols)

if symbol == "(All)":
    symbol = None

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.experimental_rerun()


# ============================================================
# 5. Load data
# ============================================================

df_trades = fetch_trades(symbol, selected_date)
df_shadow = fetch_shadow(symbol, selected_date)


# ============================================================
# 6. Daily P&L Summary
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades for this day.")
else:

    # ---- Correct Exit Detection ----
    def is_exit_flag(val):
        return val in [True, "true", "True", "1", 1, "yes", "t", "T"]

    df_trades["is_exit_norm"] = df_trades["is_exit"].apply(is_exit_flag)

    exits = df_trades[df_trades["is_exit_norm"] == True].copy()

    pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else "pnl"
    total_pnl = exits[pnl_col].fillna(0).sum()

    wins = exits["win"].fillna(False).sum() if "win" in exits else 0
    total_exits = len(exits)
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades (exits)", total_exits)
    c2.metric("Total P&L", f"{total_pnl:,.2f}")
    c3.metric("Wins", int(wins))
    c4.metric("Win Rate", f"{win_rate:.1f}%")


    # ---- Debug Panel (Shows EXITS found) ----
    with st.expander("🔍 DEBUG — Exit rows detected"):
        st.write(exits)


    # ---- P&L by Symbol ----
    st.markdown("### 💰 P&L by Symbol")

    if exits.empty:
        st.info("No exit trades — P&L unavailable.")
    else:
        pnl_by_sym = (
            exits.groupby("symbol")[pnl_col]
            .sum()
            .reset_index()
            .rename(columns={pnl_col: "total_pnl"})
        )

        st.bar_chart(pnl_by_sym.set_index("symbol")["total_pnl"])


# ============================================================
# 7. Raw Trades Table
# ============================================================

st.subheader("📜 Raw Trades (colored by P&L)")

if df_trades.empty:
    st.info("No trades recorded.")
else:
    df_disp = df_trades.sort_values("ts", ascending=False)

    def style_pnl(val):
        if val is None:
            return ""
        try:
            v = float(val)
            if v > 0:
                return "background-color: rgba(0,255,0,0.18)"
            if v < 0:
                return "background-color: rgba(255,0,0,0.18)"
        except:
            return ""

    pnl_cols = [c for c in ["pnl", "realized_pnl"] if c in df_disp.columns]

    st.dataframe(
        df_disp.style.applymap(style_pnl, subset=pnl_cols),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 8. Shadow Logs
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

    st.dataframe(
        df_shadow.sort_values("ts", ascending=False)[cols],
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("Powered by Supabase • GPT5-Trade • ML Shadow Mode • Streamlit Dashboard")
