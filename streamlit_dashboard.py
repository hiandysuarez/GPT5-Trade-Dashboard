import os
from datetime import datetime, date
from typing import Optional, List

import streamlit as st
import pandas as pd
from supabase import create_client, Client


# ============================================================
# 1. Supabase Client Setup
# ============================================================

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    """
    Initialize and cache Supabase client using Streamlit secrets.
    Expects:
        SUPABASE_URL
        SUPABASE_KEY
    in .streamlit/secrets.toml
    """
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error("Supabase secrets missing. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml.")
        st.stop()

    try:
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        return None


sb = get_supabase_client()
if sb is None:
    st.stop()
# --- Debug: show that secrets + client resolved ---
with st.sidebar.expander("🔍 Debug info", expanded=False):
    st.write("Supabase URL:", st.secrets.get("SUPABASE_URL", "missing")[:40] + "...")

# ============================================================
# 2. Basic Page Config
# ============================================================

st.set_page_config(
    page_title="GPT5-Trade Dashboard",
    layout="wide",
)

st.title("📊 GPT5-Trade Monitoring Dashboard")
st.caption("Realtime view for live trades, P&L, and ML shadow predictions")


# ============================================================
# 3. Data Fetch Helpers
# ============================================================

@st.cache_data(ttl=30)
def fetch_trades(symbol: Optional[str] = None, day: Optional[date] = None) -> pd.DataFrame:
    """
    Fetch trades from Supabase `trades` table.
    Optional filters:
        - symbol
        - specific date (UTC)
    """
    try:
        query = sb.table("trades").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        resp = query.order("ts", desc=False).execute()
        rows = resp.data or []
        df = pd.DataFrame(rows)

        # 🔍 Debug: show how many rows we fetched
        st.sidebar.write(f"Trades fetched: {len(df)}")

        return df

    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_shadow_logs(symbol: Optional[str] = None, day: Optional[date] = None) -> pd.DataFrame:
    """
    Fetch ML shadow-mode logs from Supabase `ml_shadow_logs` table.
    """
    try:
        query = sb.table("ml_shadow_logs").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        resp = query.order("ts", desc=False).execute()
        rows = resp.data or []
        df = pd.DataFrame(rows)

        # 🔍 Debug: show how many shadow rows we fetched
        st.sidebar.write(f"Shadow logs fetched: {len(df)}")

        return df

    except Exception as e:
        st.error(f"Error fetching shadow logs: {e}")
        return pd.DataFrame()


# ============================================================
# 4. Sidebar Controls
# ============================================================

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date (UTC)", value=today)

# Fetch symbols for dropdown
symbol_options: List[str] = []
trades_for_list = fetch_trades(day=selected_date)

if not trades_for_list.empty and "symbol" in trades_for_list.columns:
    symbol_options = sorted(trades_for_list["symbol"].dropna().unique().tolist())

selected_symbol = None
if symbol_options:
    selected_symbol = st.sidebar.selectbox(
        "Symbol",
        options=["(All)"] + symbol_options,
        index=0,
    )
    if selected_symbol == "(All)":
        selected_symbol = None

# Manual refresh
if st.sidebar.button("🔄 Refresh data"):
    fetch_trades.clear()
    fetch_shadow_logs.clear()
    st.experimental_rerun()


# ============================================================
# 5. Load Data With Filters
# ============================================================

df_trades = fetch_trades(symbol=selected_symbol, day=selected_date)
df_shadow = fetch_shadow_logs(symbol=selected_symbol, day=selected_date)

# Normalize timestamps
for col in ("ts",):
    if col in df_trades.columns:
        df_trades[col] = pd.to_datetime(df_trades[col], errors="coerce")
    if col in df_shadow.columns:
        df_shadow[col] = pd.to_datetime(df_shadow[col], errors="coerce")


# ============================================================
# 6. Top-Level Metrics
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades found for this date / filters.")
else:
    exits = df_trades[df_trades["is_exit"] == True].copy()

    total_trades = len(exits)
    total_pnl = float(exits.get("realized_pnl", exits.get("pnl", 0)).fillna(0).sum())
    wins = int(exits["win"].fillna(False).sum()) if "win" in exits.columns else 0
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Total P&L", f"{total_pnl:,.2f}")
    col3.metric("Wins", wins)
    col4.metric("Win Rate", f"{win_rate:.1f}%")

    # PNL by symbol
    st.markdown("#### P&L by Symbol")

    if "symbol" in exits.columns:
        pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else "pnl"

        pnl_by_sym = (
            exits.groupby("symbol")[pnl_col]
            .sum()
            .reset_index()
            .rename(columns={pnl_col: "total_pnl"})
        )

        st.bar_chart(
            pnl_by_sym.set_index("symbol")["total_pnl"],
            height=220,
        )


# ============================================================
# 7. Real Trades Table
# ============================================================

st.subheader("📜 Raw Trades")

if df_trades.empty:
    st.write("No trade rows.")
else:
    df_disp = df_trades.sort_values("ts", ascending=False)

    cols = [
        "ts", "symbol", "side", "qty", "fill_price",
        "pnl", "pnl_pct",
        "realized_pnl", "realized_pnl_pct",
        "win", "is_entry", "is_exit",
        "exit_reason", "confidence", "reasoning",
        "order_id", "entry_trade_id", "model_version",
    ]
    cols = [c for c in cols if c in df_disp.columns]

    st.dataframe(df_disp[cols], use_container_width=True, hide_index=True)


# ============================================================
# 8. Shadow vs Real
# ============================================================

st.subheader("🧪 ML Shadow-Mode vs Real Trades")

if df_shadow.empty:
    st.info("No shadow-mode rows.")
else:
    st.markdown("#### Shadow Overview")

    total_shadow = len(df_shadow)
    symbol_count = df_shadow["symbol"].dropna().nunique() if "symbol" in df_shadow.columns else 0

    c1, c2 = st.columns(2)
    c1.metric("Shadow Logs", total_shadow)
    c2.metric("Symbols", symbol_count)

    if "ml_direction" in df_shadow.columns:
        dir_counts = (
            df_shadow["ml_direction"]
            .fillna("UNKNOWN")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "direction", "ml_direction": "count"})
        )

        st.markdown("#### Shadow Prediction Directions")
        st.bar_chart(dir_counts.set_index("direction")["count"], height=220)

    st.markdown("#### Shadow Raw Logs")

    pref_cols = [
        "ts", "symbol", "ml_direction", "ml_probs",
        "ml_win_prob", "bot_action", "real_trade_id"
    ]
    present = [c for c in pref_cols if c in df_shadow.columns]

    df_shadow_disp = df_shadow.sort_values("ts", ascending=False)

    st.dataframe(df_shadow_disp[present], use_container_width=True, hide_index=True)


# ============================================================
# 9. Footer
# ============================================================

st.markdown("---")
st.caption("Powered by Supabase • GPT5-Trade Shadow Mode • Streamlit Dashboard")
