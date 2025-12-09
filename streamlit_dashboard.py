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
    """Initialize Supabase client from Streamlit secrets."""
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
# 2. Utilities
# ============================================================

def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ts column to datetime."""
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def normalize_exit_flag(val) -> bool:
    """Robust is_exit normalization: handles booleans, ints, strings."""
    if val is True:
        return True
    if val in [False, None]:
        return False

    s = str(val).strip().lower()
    return s in ("true", "1", "t", "yes", "y")


# ============================================================
# 3. Page Config
# ============================================================

st.set_page_config(
    page_title="GPT5-Trade Dashboard",
    layout="wide",
)

st.title("📊 GPT5-Trade Monitoring Dashboard")
st.caption("Realtime view for entries, exits, P&L, and ML shadow predictions")


# ============================================================
# 4. Fetch Helpers
# ============================================================

@st.cache_data(ttl=10)
def fetch_trades(symbol: Optional[str], day: Optional[date]) -> pd.DataFrame:
    """Fetch trades from Supabase `trades` table."""
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
def fetch_shadow(symbol: Optional[str], day: Optional[date]) -> pd.DataFrame:
    """Fetch shadow-mode logs from `ml_shadow_logs`."""
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
# 5. Sidebar
# ============================================================

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date", today)

df_all_for_symbols = fetch_trades(None, selected_date)

symbols = (
    sorted(df_all_for_symbols["symbol"].dropna().unique().tolist())
    if "symbol" in df_all_for_symbols.columns
    else []
)
symbol = st.sidebar.selectbox("Symbol", ["(All)"] + symbols)

if symbol == "(All)":
    symbol = None

if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.experimental_rerun()


# ============================================================
# 6. Load data (date + symbol filtered)
# ============================================================

df_trades = fetch_trades(symbol, selected_date)
df_shadow = fetch_shadow(symbol, selected_date)


# ============================================================
# 7. Daily Performance Section
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades for this day.")
else:
    # Normalize exit detection fully
    df_trades["is_exit_norm"] = df_trades["is_exit"].apply(normalize_exit_flag)

    exits = df_trades[df_trades["is_exit_norm"] == True].copy()

    pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else "pnl"
    total_pnl = exits[pnl_col].fillna(0).sum()

    wins = exits["win"].fillna(False).sum() if "win" in exits.columns else 0
    total_exits = len(exits)
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades (exits)", total_exits)
    c2.metric("Total P&L", f"{total_pnl:,.2f}")
    c3.metric("Wins", int(wins))
    c4.metric("Win Rate", f"{win_rate:.1f}%")

    # -------- P&L by Symbol (static, adaptive range) --------
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

        if pnl_by_sym.empty:
            st.info("No P&L data by symbol.")
        else:
            # Thinner bars & static chart via altair-like config
            chart_data = pnl_by_sym.set_index("symbol")["total_pnl"]

            # Streamlit's bar_chart is simple, but we can at least ensure it draws properly
            st.bar_chart(chart_data)


# ============================================================
# 8. Raw Trades Table (Global Last 10 Exits, Color-Coded)
# ============================================================

st.subheader("📜 Latest Exit Trades (Global, last 10, colored by P&L)")

# Fetch ALL trades ignoring date filter
df_all_trades = fetch_trades(symbol=None, day=None)

if df_all_trades.empty:
    st.info("No recorded trades exist.")
else:
    df_all_trades = normalize_ts(df_all_trades)
    df_all_trades["is_exit_norm"] = df_all_trades["is_exit"].apply(normalize_exit_flag)

    # Decide which PnL column to use
    pnl_col_global = "realized_pnl" if "realized_pnl" in df_all_trades.columns else "pnl"

    df_exits = df_all_trades[
        (df_all_trades["is_exit_norm"] == True) &
        (df_all_trades[pnl_col_global].notna())
    ].copy()

    if df_exits.empty:
        st.info("No exit trades with P&L yet.")
    else:
        # Sort newest first and take last 10
        df_exits = df_exits.sort_values("ts", ascending=False).head(10)

        # Color function for P&L
        def color_pnl(val):
            try:
                v = float(val)
                if v > 0:
                    return "background-color: rgba(0, 255, 0, 0.25);"  # green
                if v < 0:
                    return "background-color: rgba(255, 0, 0, 0.25);"  # red
            except Exception:
                return ""
            return ""

        # Use Styler.applymap (works today; ignore future deprecation warning)
        styled_exits = df_exits.style.applymap(color_pnl, subset=[pnl_col_global])

        st.dataframe(
            styled_exits,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# 9. Shadow Logs Section
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
