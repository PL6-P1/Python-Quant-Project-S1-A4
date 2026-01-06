# app/single_asset/ui.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

# Auto-refresh every 5 minutes
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return None


TRADING_DAYS = 252


# ============================
#   Helpers / Metrics
# ============================

def to_returns(price: pd.Series) -> pd.Series:
    return price.pct_change().fillna(0.0)


def equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods))


def annualized_return(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    if equity.empty or len(equity) < 2:
        return np.nan
    total = float(equity.iloc[-1] / equity.iloc[0])
    n = len(equity) - 1
    return float(total ** (periods / n) - 1.0)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    if returns.empty:
        return np.nan
    rf_per = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = returns - rf_per
    denom = excess.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.sqrt(periods) * excess.mean() / denom)


def compute_metrics(equity: pd.Series, strat_returns: pd.Series) -> Dict[str, float]:
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    strat_returns = strat_returns.replace([np.inf, -np.inf], np.nan).dropna()

    if equity.empty or strat_returns.empty:
        return {
            "Total return": np.nan,
            "Annualized return": np.nan,
            "Annualized vol": np.nan,
            "Sharpe": np.nan,
            "Max drawdown": np.nan,
        }

    total_return = float(equity.iloc[-1] - 1.0)
    ann_ret = annualized_return(equity)
    ann_vol = annualized_vol(strat_returns)
    sharpe = sharpe_ratio(strat_returns, rf_annual=0.0)
    mdd = max_drawdown(equity)

    return {
        "Total return": total_return,
        "Annualized return": ann_ret,
        "Annualized vol": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": mdd,
    }


# ============================
#   Plot helpers (Plotly)
# ============================

def plot_series_plotly(title: str, s: pd.Series, y_label: str = "Value") -> None:
    """
    Robust plot for constant / near-constant series (Streamlit/Altair line_chart can look blank).
    """
    if s is None or len(s) == 0:
        st.warning(f"{title}: no data.")
        return

    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s2.empty:
        st.warning(f"{title}: empty after cleaning.")
        return

    y_min = float(s2.min())
    y_max = float(s2.max())

    # If constant, expand range slightly to show a visible line
    if np.isfinite(y_min) and np.isfinite(y_max) and abs(y_max - y_min) < 1e-12:
        pad = max(0.01 * abs(y_min), 1e-3)
        y_min -= pad
        y_max += pad
    else:
        pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s2.index, y=s2.values, mode="lines", name=title))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        yaxis=dict(range=[y_min, y_max]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_df_plotly(title: str, df: pd.DataFrame, y_label: str = "Value") -> None:
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan)

    # Keep rows where at least one column has data
    df2 = df2.dropna(how="all")
    if df2.empty:
        st.warning(f"{title}: no data to plot.")
        return

    fig = go.Figure()
    for col in df2.columns:
        fig.add_trace(go.Scatter(x=df2.index, y=df2[col], mode="lines", name=str(col)))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


@dataclass
class StrategyOutput:
    name: str
    equity: pd.Series
    returns: pd.Series


def find_output(outputs: List[StrategyOutput], startswith: str) -> Optional[StrategyOutput]:
    for o in outputs:
        if o.name.startswith(startswith):
            return o
    return None


# ============================
#   Strategies
# ============================

def strat_buy_hold(price: pd.Series) -> StrategyOutput:
    r = to_returns(price)
    eq = equity_from_returns(r)
    return StrategyOutput("Buy & Hold", eq, r)


def strat_sma_crossover(price: pd.Series, short_window: int, long_window: int) -> StrategyOutput:
    ma_s = price.rolling(short_window).mean()
    ma_l = price.rolling(long_window).mean()
    signal = (ma_s > ma_l).astype(float).shift(1).fillna(0.0)

    r = to_returns(price)
    strat_r = r * signal
    eq = equity_from_returns(strat_r)
    return StrategyOutput(f"SMA Crossover ({short_window},{long_window})", eq, strat_r)


def _rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def strat_rsi_mean_reversion(price: pd.Series, rsi_window: int, low: int, high: int) -> StrategyOutput:
    rsi = _rsi(price, rsi_window)

    position = pd.Series(0.0, index=price.index)
    position[rsi < low] = 1.0
    position[rsi > high] = 0.0
    position = position.shift(1).fillna(0.0)

    r = to_returns(price)
    strat_r = r * position
    eq = equity_from_returns(strat_r)

    return StrategyOutput(f"RSI Mean Reversion (w={rsi_window}, {low}/{high})", eq, strat_r)


def strat_bollinger_mean_reversion(price: pd.Series, window: int, n_std: float) -> StrategyOutput:
    ma = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd

    position = pd.Series(0.0, index=price.index)
    position[price < lower] = 1.0
    position[price > ma] = 0.0
    position = position.shift(1).fillna(0.0)

    r = to_returns(price)
    strat_r = r * position
    eq = equity_from_returns(strat_r)

    return StrategyOutput(f"Bollinger MR (w={window}, sigma={n_std:.1f})", eq, strat_r)


def strat_donchian_breakout(price: pd.Series, window: int) -> StrategyOutput:
    hi = price.rolling(window).max().shift(1)
    lo = price.rolling(window).min().shift(1)

    position = pd.Series(0.0, index=price.index)
    in_pos = False

    for i in range(len(price)):
        c = float(price.iloc[i])
        h = hi.iloc[i]
        l = lo.iloc[i]

        if pd.isna(h) or pd.isna(l):
            position.iloc[i] = 0.0
            continue

        if (not in_pos) and (c > float(h)):
            in_pos = True
        elif in_pos and (c < float(l)):
            in_pos = False

        position.iloc[i] = 1.0 if in_pos else 0.0

    position = position.shift(1).fillna(0.0)

    r = to_returns(price)
    strat_r = r * position
    eq = equity_from_returns(strat_r)

    return StrategyOutput(f"Donchian Breakout (w={window})", eq, strat_r)


# ============================
#   Data loader
# ============================

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _select_price_series(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    elif "Close" in df.columns:
        s = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' column found.")
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


# ============================
#   Streamlit page (Quant A)
# ============================

def run_single_asset():
    st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

    st.title("Single Asset Analysis (Quant A)")
    st.caption("Single-asset analysis: Yahoo Finance data, backtests, metrics, and per-strategy diagnostics.")

    st.sidebar.header("Settings")

    # --- Small ticker-format reminder (the thing you requested) ---
    st.sidebar.caption(
        "Ticker format tips:\n"
        "- French equities: use the exchange suffix, e.g. ENGI.PA, TTE.PA\n"
        "- FX pairs: use '=X', e.g. EURUSD=X, GBPUSD=X\n"
        "- Crypto: BTC-USD, ETH-USD\n"
    )

    ticker = st.sidebar.text_input("Ticker", value="META").strip().upper()

    today = date.today()
    default_start = today - timedelta(days=365)

    start = st.sidebar.date_input("Start date", value=default_start, max_value=today)
    end = st.sidebar.date_input("End date", value=today, max_value=today)

    st.sidebar.subheader("Strategies")
    available = [
        "Buy & Hold",
        "SMA Crossover",
        "RSI Mean Reversion",
        "Bollinger Mean Reversion",
        "Donchian Breakout",
    ]
    selected = st.sidebar.multiselect("Select strategies", options=available, default=available)

    st.sidebar.subheader("SMA Crossover params")
    short_window = st.sidebar.slider("Short SMA (days)", 5, 50, 20, 1)
    long_window = st.sidebar.slider("Long SMA (days)", 20, 200, 50, 5)

    st.sidebar.subheader("RSI params")
    rsi_window = st.sidebar.slider("RSI window", 5, 50, 14, 1)
    rsi_low = st.sidebar.slider("Oversold threshold", 5, 45, 30, 1)
    rsi_high = st.sidebar.slider("Overbought threshold", 55, 95, 70, 1)

    st.sidebar.subheader("Bollinger params")
    bb_window = st.sidebar.slider("BB window", 10, 100, 20, 1)
    bb_nstd = st.sidebar.slider("BB std (σ)", 1.0, 3.0, 1.5, 0.1)

    st.sidebar.subheader("Donchian params")
    don_window = st.sidebar.slider("Donchian window", 10, 200, 50, 5)

    # Validation
    if not ticker:
        st.error("Please enter a valid ticker (e.g., META, AAPL, ENGI.PA, EURUSD=X).")
        return
    if start >= end:
        st.error("Start date must be strictly before end date.")
        return
    if short_window >= long_window:
        st.error("Short SMA must be strictly smaller than Long SMA.")
        return
    if rsi_low >= rsi_high:
        st.error("RSI oversold threshold must be smaller than overbought threshold.")
        return
    if len(selected) == 0:
        st.warning("Select at least one strategy.")
        return

    # Load data
    with st.spinner(f"Downloading data for {ticker}..."):
        df = load_ohlcv(ticker, start, end)

    if df.empty:
        st.warning(
            "No data found for this ticker / date range.\n\n"
            "Reminder: FR equities often need '.PA' (e.g., ENGI.PA). FX needs '=X' (e.g., EURUSD=X)."
        )
        return

    try:
        price = _select_price_series(df)
    except ValueError as e:
        st.error(str(e))
        return

    min_len_needed = max(long_window + 5, bb_window + 5, don_window + 5, rsi_window + 5)
    if len(price) < min_len_needed:
        st.warning("Date range too short for current parameters. Increase the period or reduce windows.")
        return

    # Compute strategies
    outputs: List[StrategyOutput] = []
    if "Buy & Hold" in selected:
        outputs.append(strat_buy_hold(price))
    if "SMA Crossover" in selected:
        outputs.append(strat_sma_crossover(price, short_window, long_window))
    if "RSI Mean Reversion" in selected:
        outputs.append(strat_rsi_mean_reversion(price, rsi_window, rsi_low, rsi_high))
    if "Bollinger Mean Reversion" in selected:
        outputs.append(strat_bollinger_mean_reversion(price, bb_window, bb_nstd))
    if "Donchian Breakout" in selected:
        outputs.append(strat_donchian_breakout(price, don_window))

    # Equity dataframe (do NOT drop rows where only some strategies are NaN)
    equity_df = pd.concat([o.equity.rename(o.name) for o in outputs], axis=1)
    equity_df = equity_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    # Price normalized (aligned on price index)
    price_norm = (price / price.iloc[0]).rename("Price (normalized)")

    # Combine without killing the price column
    combined_df = pd.concat([price_norm, equity_df], axis=1)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    metrics_rows = {o.name: compute_metrics(o.equity, o.returns) for o in outputs}
    metrics_df = pd.DataFrame(metrics_rows).T

    tab_overview, tab_strat, tab_data = st.tabs(["Overview", "Strategies", "Data"])

    with tab_overview:
        last_price = float(price.iloc[-1])
        if len(price) > 1:
            prev = float(price.iloc[-2])
            delta_value = last_price - prev
            delta_pct = (last_price / prev - 1.0) * 100.0
            st.metric("Last price", f"{last_price:.2f}", delta=f"{delta_value:.2f} ({delta_pct:+.2f}%)")
        else:
            st.metric("Last price", f"{last_price:.2f}")

        st.subheader("Main chart: normalized price vs strategies")
        # This should always show the Price (normalized) now
        plot_df_plotly("Normalized price vs strategies", combined_df, y_label="Normalized value")

        st.subheader("Performance metrics")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Total return": "{:.2%}",
                    "Annualized return": "{:.2%}",
                    "Annualized vol": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max drawdown": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "Note: Buy & Hold often beats simple rule-based strategies on trending assets over short horizons (e.g., 1Y). "
            "This is not necessarily a bug; it depends on regime (trend vs mean reversion)."
        )

    with tab_strat:
        st.subheader("Equity curves (backtests)")
        plot_df_plotly("Equity curves", equity_df, y_label="Equity")

        st.divider()
        st.subheader("Per-strategy diagnostics")

        if "SMA Crossover" in selected:
            with st.expander("SMA Crossover diagnostics", expanded=True):
                ma_s = price.rolling(short_window).mean().rename(f"SMA {short_window}")
                ma_l = price.rolling(long_window).mean().rename(f"SMA {long_window}")
                plot_df_plotly("Price + SMAs", pd.concat([price.rename("Price"), ma_s, ma_l], axis=1), y_label="Price")

                pos = (ma_s > ma_l).astype(float).shift(1).fillna(0.0)
                plot_df_plotly("Position (1 = long; shifted by 1 day)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "SMA Crossover")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

        if "RSI Mean Reversion" in selected:
            with st.expander("RSI Mean Reversion diagnostics", expanded=False):
                rsi = _rsi(price, rsi_window)
                rsi_df = pd.DataFrame(
                    {"RSI": rsi, "Oversold": float(rsi_low), "Overbought": float(rsi_high)},
                    index=rsi.index,
                )
                plot_df_plotly("RSI + thresholds", rsi_df, y_label="RSI")

                pos = pd.Series(0.0, index=price.index)
                pos[rsi < rsi_low] = 1.0
                pos[rsi > rsi_high] = 0.0
                pos = pos.shift(1).fillna(0.0)
                plot_df_plotly("Position (1 = long; shifted by 1 day)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "RSI Mean Reversion")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

        if "Bollinger Mean Reversion" in selected:
            with st.expander("Bollinger Mean Reversion diagnostics", expanded=False):
                ma = price.rolling(bb_window).mean().rename("Middle")
                sd = price.rolling(bb_window).std(ddof=0)
                upper = (ma + bb_nstd * sd).rename("Upper")
                lower = (ma - bb_nstd * sd).rename("Lower")

                plot_df_plotly(
                    "Price + Bollinger Bands",
                    pd.concat([price.rename("Price"), ma, upper, lower], axis=1),
                    y_label="Price",
                )

                pos = pd.Series(0.0, index=price.index)
                pos[price < lower] = 1.0
                pos[price > ma] = 0.0
                pos = pos.shift(1).fillna(0.0)
                plot_df_plotly("Position (1 = long; shifted by 1 day)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "Bollinger MR")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

        if "Donchian Breakout" in selected:
            with st.expander("Donchian Breakout diagnostics", expanded=False):
                hi = price.rolling(don_window).max().shift(1).rename("Channel High")
                lo = price.rolling(don_window).min().shift(1).rename("Channel Low")
                plot_df_plotly("Price + Donchian Channel", pd.concat([price.rename("Price"), hi, lo], axis=1), y_label="Price")

                pos = pd.Series(0.0, index=price.index)
                in_pos = False
                for i in range(len(price)):
                    c = float(price.iloc[i])
                    h = hi.iloc[i]
                    l = lo.iloc[i]
                    if pd.isna(h) or pd.isna(l):
                        pos.iloc[i] = 0.0
                        continue
                    if (not in_pos) and (c > float(h)):
                        in_pos = True
                    elif in_pos and (c < float(l)):
                        in_pos = False
                    pos.iloc[i] = 1.0 if in_pos else 0.0
                pos = pos.shift(1).fillna(0.0)

                plot_df_plotly("Position (1 = long; shifted by 1 day)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "Donchian Breakout")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

    with tab_data:
        st.subheader("OHLCV data (tail)")
        st.dataframe(df.tail(20), use_container_width=True)
        st.caption("Source: Yahoo Finance (yfinance). Cached 5 minutes (TTL=300s).")
