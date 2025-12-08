import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List

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
except Exception as e:
st.error("Supabase secrets missing. Set SUPABASE_URL and SUPABASE_KEY in secrets.toml.")
st.stop()

```
try:
    client = create_client(url, key)
    return client
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}")
    return None
```

sb = get_supabase_client()
if sb is None:
st.stop()

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

```
    if symbol:
        query = query.eq("symbol", symbol)

    if day:
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

    resp = query.order("ts", desc=False).execute()
    rows = resp.data or []
    df = pd.DataFrame(rows)
    return df
except Exception as e:
    st.error(f"Error fetching trades: {e}")
    return pd.DataFrame()
```

@st.cache_data(ttl=30)
def fetch_shadow_logs(symbol: Optional[str] = None, day: Optional[date] = None) -> pd.DataFrame:
"""
Fetch ML shadow-mode logs from Supabase `ml_shadow_logs` table.
"""
try:
query = sb.table("ml_shadow_logs").select("*")

```
    if symbol:
        query = query.eq("symbol", symbol)

    if day:
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        query = query.gte("ts", start.isoformat()).lte("ts", end.isoformat())

    resp = query.order("ts", desc=False).execute()
    rows = resp.data or []
    df = pd.DataFrame(rows)
    return df
except Exception as e:
    st.error(f"Error fetching shadow logs: {e}")
    return pd.DataFrame()
```

# ============================================================

# 4. Sidebar Controls

# ============================================================

st.sidebar.header("Filters")

# Date filter (default = today)

today = datetime.utcnow().date()
selected_date = st.sidebar.date_input("Trading date (UTC)", value=today)

# Symbol filter

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
# Clear cached data
fetch_trades.clear()
fetch_shadow_logs.clear()
st.experimental_rerun()

# ============================================================

# 5. Load Data With Filters

# ============================================================

df_trades = fetch_trades(symbol=selected_symbol, day=selected_date)
df_shadow = fetch_shadow_logs(symbol=selected_symbol, day=selected_date)

# Normalize timestamp columns to datetime

for col in ("ts",):
if col in df_trades.columns:
df_trades[col] = pd.to_datetime(df_trades[col], errors="coerce")
if col in df_shadow.columns:
df_shadow[col] = pd.to_datetime(df_shadow[col], errors="coerce")

# ============================================================

# 6. Top-Level Metrics (Real Trades)

# ============================================================

st.subheader("📈 Daily Performance — Real Trades")

if df_trades.empty:
st.info("No trades found for this date / filters.")
else:
# Only exits (round-trips show at exits)
exits = df_trades[df_trades["is_exit"] == True].copy()

```
total_trades = len(exits)
total_pnl = float(exits["realized_pnl"].fillna(0).sum()) if "realized_pnl" in exits.columns else float(exits["pnl"].fillna(0).sum())
wins = int(exits["win"].fillna(False).sum()) if "win" in exits.columns else 0
losses = total_trades - wins
win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total Trades (exits)", total_trades)
col_b.metric("P&L (realized)", f"{total_pnl:,.2f}")
col_c.metric("Wins", wins)
col_d.metric("Win Rate %", f"{win_rate:.1f}%")


# --------------------------------------------------------
# P&L by Symbol
# --------------------------------------------------------
st.markdown("#### P&L by Symbol")

if "symbol" in exits.columns:
    if "realized_pnl" in exits.columns:
        pnl_col = "realized_pnl"
    else:
        pnl_col = "pnl"

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
else:
    st.write("No symbol column available in trades table.")
```

# ============================================================

# 7. Real Trades Table

# ============================================================

st.subheader("📜 Raw Trades (entries & exits)")

if df_trades.empty:
st.write("No trade rows to display.")
else:
# Sort newest first
df_trades_disp = df_trades.sort_values("ts", ascending=False).copy()

```
# Select subset of useful columns if present
cols_preferred = [
    "ts",
    "symbol",
    "side",
    "qty",
    "fill_price",
    "pnl",
    "pnl_pct",
    "realized_pnl",
    "realized_pnl_pct",
    "win",
    "is_entry",
    "is_exit",
    "exit_reason",
    "confidence",
    "reasoning",
    "order_id",
    "entry_trade_id",
    "model_version",
]
cols_present = [c for c in cols_preferred if c in df_trades_disp.columns]
if cols_present:
    st.dataframe(
        df_trades_disp[cols_present],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(df_trades_disp, use_container_width=True, hide_index=True)
```

# ============================================================

# 8. Shadow vs Real — Comparison

# ============================================================

st.subheader("🧪 ML Shadow-Mode vs Real Trades")

if df_shadow.empty:
st.info("No shadow-mode logs found for this date / filters.")
else:
# Basic overview
st.markdown("#### Shadow-mode Overview")

```
total_shadow = len(df_shadow)
unique_shadow_symbols = df_shadow["symbol"].dropna().nunique() if "symbol" in df_shadow.columns else 0

c1, c2 = st.columns(2)
c1.metric("Shadow logs", total_shadow)
c2.metric("Symbols (shadow)", unique_shadow_symbols)

# If ml_direction + ml_win_prob exist, show distribution
if "ml_direction" in df_shadow.columns:
    dir_counts = (
        df_shadow["ml_direction"]
        .fillna("UNKNOWN")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "direction", "ml_direction": "count"})
    )

    st.markdown("#### Shadow prediction directions")
    st.bar_chart(
        dir_counts.set_index("direction")["count"],
        height=220,
    )

# Show raw shadow logs (selected columns)
st.markdown("#### Raw shadow logs")
cols_shadow_pref = [
    "ts",
    "symbol",
    "ml_direction",
    "ml_probs",
    "ml_win_prob",
    "bot_action",
    "real_trade_id",
]
cols_shadow_present = [c for c in cols_shadow_pref if c in df_shadow.columns]

df_shadow_disp = df_shadow.sort_values("ts", ascending=False).copy()

if cols_shadow_present:
    st.dataframe(
        df_shadow_disp[cols_shadow_present],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(df_shadow_disp, use_container_width=True, hide_index=True)
```

# ============================================================

# 9. Footer / Debug

# ============================================================

st.markdown("---")
st.caption(
"Backend: Supabase `trades` + `ml_shadow_logs`. "
"This dashboard is read-only and safe to expose to Streamlit Cloud."
)
