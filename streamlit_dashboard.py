import os
from datetime import datetime, date
from typing import Optional, List

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh


# ============================================================
# Auto-refresh every 60 seconds
# ============================================================
st_autorefresh(interval=60_000, key="refresh_timer")


# ============================================================
# 1. Supabase Client Setup
# ============================================================

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    """Initialize and cache Supabase client."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error("Supabase secrets missing. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml.")
        st.stop()

    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        return None


sb = get_supabase_client()
if sb is None:
    st.stop()


# ============================================================
# 2. Page Config
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
def fetch_trades(symbol: Optional[str] = None, day: Optional[date] = None):
    """Fetch trades from Supabase."""
    try:
        query = sb.table("trades").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        resp = query.order("ts", desc=False).execute()
        df = pd.DataFrame(resp.data or [])

        st.sidebar.write(f"Trades fetched: {len(df)}")
        return df

    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_shadow_logs(symbol: Optional[str] = None, day: Optional[date] = None):
    """Fetch shadow-mode logs."""
    try:
        query = sb.table("ml_shadow_logs").select("*")

        if symbol:
            query = query.eq("symbol", symbol)

        if day:
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

        resp = query.order("ts", desc=False).execute()
        df = pd.DataFrame(resp.data or [])

        st.sidebar.write(f"Shadow logs fetched: {len(df)}")
        return df

    except Exception as e:
        st.error(f"Error fetching shadow logs: {e}")
        return pd.DataFrame()


# ============================================================
# 4. Sidebar Filters
# ============================================================

st.sidebar.header("Filters")

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date (UTC)", value=today)

# Symbol dropdown (from today's trades)
symbol_options = []
df_for_list = fetch_trades(day=selected_date)

if not df_for_list.empty and "symbol" in df_for_list.columns:
    symbol_options = sorted(df_for_list["symbol"].dropna().unique().tolist())

selected_symbol = st.sidebar.selectbox(
    "Symbol",
    options=["(All)"] + symbol_options,
    index=0,
)

if selected_symbol == "(All)":
    selected_symbol = None

# Manual refresh
if st.sidebar.button("🔄 Refresh now"):
    st.cache_data.clear()
    st.experimental_rerun()


# ============================================================
# 5. Load Trades + Shadow Logs
# ============================================================

df_trades = fetch_trades(symbol=selected_symbol, day=selected_date)
df_shadow = fetch_shadow_logs(symbol=selected_symbol, day=selected_date)

# Normalize timestamps
for df in [df_trades, df_shadow]:
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")


# ============================================================
# 6. Daily Performance Section
# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
    st.info("No trades found for this date / filters.")
else:
    # Normalize exit flag FIRST
    df_trades["is_exit_norm"] = (
        df_trades["is_exit"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    )

    exits = df_trades[df_trades["is_exit_norm"] == True].copy()

    total_trades = len(exits)
    pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else "pnl"
    total_pnl = float(exits[pnl_col].fillna(0).sum())

    wins = int(exits["win"].fillna(False).sum()) if "win" in exits.columns else 0
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", total_trades)
    c2.metric("Total P&L", f"{total_pnl:,.2f}")
    c3.metric("Wins", wins)
    c4.metric("Win Rate", f"{win_rate:.1f}%")

    # ---- PNL by Symbol ----
    st.markdown("#### P&L by Symbol")

    if not exits.empty:
        pnl_by_sym = (
            exits.groupby("symbol")[pnl_col]
            .sum()
            .reset_index()
            .rename(columns={pnl_col: "total_pnl"})
        )

        st.bar_chart(pnl_by_sym.set_index("symbol")["total_pnl"], height=240)


# ============================================================
# 7. Color-coded Real Trades Table
# ============================================================

st.subheader("📜 Raw Trades (Colored by P&L)")

if df_trades.empty:
    st.info("No trades to display.")
else:
    df_disp = df_trades.sort_values("ts", ascending=False).copy()

    # Color coding
    def color_pnl(val):
        try:
            v = float(val)
            if v > 0:
                return "background-color: rgba(0,255,0,0.2);"   # green
            elif v < 0:
                return "background-color: rgba(255,0,0,0.2);"   # red
        except:
            pass
        return ""

    cols = [
        "ts", "symbol", "side", "qty", "fill_price",
        "pnl", "pnl_pct", "realized_pnl", "realized_pnl_pct",
        "win", "is_entry", "is_exit", "exit_reason",
        "confidence", "reasoning", "order_id", "entry_trade_id"
    ]

    cols = [c for c in cols if c in df_disp.columns]

    styled = df_disp[cols].style.applymap(color_pnl, subset=["pnl", "realized_pnl"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================
# 8. Shadow vs Real
# ============================================================

st.subheader("🧪 ML Shadow-Mode vs Real Trades")

if df_shadow.empty:
    st.info("No shadow logs available.")
else:
    st.markdown("#### Shadow Predictions Overview")

    c1, c2 = st.columns(2)
    c1.metric("Shadow Logs", len(df_shadow))
    c2.metric("Symbols Logged", df_shadow["symbol"].nunique())

    if "ml_direction" in df_shadow.columns:
        counts = (
            df_shadow["ml_direction"]
            .fillna("UNKNOWN")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "direction", "ml_direction": "count"})
        )

        st.bar_chart(counts.set_index("direction")["count"], height=240)

    st.markdown("#### Raw Shadow Logs")
    cols_shadow = [
        "ts", "symbol", "ml_direction", "ml_probs",
        "ml_win_prob", "bot_action", "real_trade_id"
    ]
    cols_shadow = [c for c in cols_shadow if c in df_shadow.columns]

    st.dataframe(
        df_shadow.sort_values("ts", ascending=False)[cols_shadow],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("Powered by Supabase • GPT5-Trade Shadow Mode • Streamlit Dashboard")
