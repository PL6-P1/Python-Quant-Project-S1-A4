# app/single_asset/ui.py
"""
Quant A — Single Asset Analysis Module (Univariate)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import time as pytime


# ----------------------------
# Auto-refresh: rerun the app every 5 minutes
# ----------------------------
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return None


TRADING_DAYS = 252  # used for annualization when interval is daily


# ============================
# Helpers / Returns / Metrics
# ============================

def to_returns(price: pd.Series) -> pd.Series:
    """Simple returns from a price series."""
    return price.pct_change().fillna(0.0)


def equity_from_returns(returns: pd.Series) -> pd.Series:
    """Equity curve starting at 1.0 from a returns series."""
    return (1.0 + returns).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (min of equity/peak - 1)."""
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def annualized_vol(returns: pd.Series, periods: int) -> float:
    """Annualized volatility using sqrt(periods)."""
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=0) * np.sqrt(periods))


def annualized_return(equity: pd.Series, periods: int) -> float:
    """Annualized return based on equity ratio and number of bars."""
    if equity.empty or len(equity) < 2:
        return np.nan
    total = float(equity.iloc[-1] / equity.iloc[0])
    n = len(equity) - 1
    if n <= 0 or total <= 0:
        return np.nan
    return float(total ** (periods / n) - 1.0)


def sharpe_ratio(returns: pd.Series, rf_annual: float, periods: int) -> float:
    """Sharpe ratio with optional annual risk-free rate."""
    if returns.empty:
        return np.nan
    rf_per = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = returns - rf_per
    denom = excess.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.sqrt(periods) * excess.mean() / denom)


def compute_metrics(equity: pd.Series, strat_returns: pd.Series, periods: int) -> Dict[str, float]:
    """Compute simple performance metrics on the *already computed* equity and returns."""
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

    return {
        "Total return": float(equity.iloc[-1] - 1.0),
        "Annualized return": annualized_return(equity, periods=periods),
        "Annualized vol": annualized_vol(strat_returns, periods=periods),
        "Sharpe": sharpe_ratio(strat_returns, rf_annual=0.0, periods=periods),
        "Max drawdown": max_drawdown(equity),
    }


# ============================
# Plot helpers (Plotly)
# ============================

def plot_series_plotly(title: str, s: pd.Series, y_label: str = "Value") -> None:
    """Plot a single time series with a safe y-range (avoid flat-line range issues)."""
    if s is None or len(s) == 0:
        st.warning(f"{title}: no data.")
        return

    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s2.empty:
        st.warning(f"{title}: empty after cleaning.")
        return

    y_min = float(s2.min())
    y_max = float(s2.max())

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
    """Plot multiple columns of a DataFrame as separate lines."""
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(how="all")

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


def plot_forecast_plotly(title: str, hist: pd.Series, fcst: pd.DataFrame, y_label: str = "Price") -> None:
    """
    Plot historical series + forecast with confidence band.

    Expected fcst columns:
    - yhat
    - yhat_lower
    - yhat_upper

    Note:
    - fcst is expected (by design) to include an anchor row at hist.index[-1]
      so the dashed line starts exactly from the last observed price.
    """
    if hist is None or hist.empty or fcst is None or fcst.empty:
        st.warning("Forecast plot: missing data.")
        return

    hist2 = pd.to_numeric(hist, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fcst2 = fcst.copy()
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        if c in fcst2.columns:
            fcst2[c] = pd.to_numeric(fcst2[c], errors="coerce")
    fcst2 = fcst2.replace([np.inf, -np.inf], np.nan).dropna()

    if hist2.empty or fcst2.empty:
        st.warning("Forecast plot: empty after cleaning.")
        return

    fcst2 = fcst2.sort_index()

    # High-contrast palette for readability
    col_hist = "#1f77b4"              # blue
    col_fcst = "#d62728"              # red
    col_ci = "rgba(214,39,40,0.16)"   # transparent red

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist2.index,
        y=hist2.values,
        mode="lines",
        name="Historical",
        line=dict(color=col_hist, width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=fcst2.index,
        y=fcst2["yhat_upper"].values,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fcst2.index,
        y=fcst2["yhat_lower"].values,
        mode="lines",
        fill="tonexty",
        fillcolor=col_ci,
        line=dict(color="rgba(0,0,0,0)", width=0),
        name="CI",
        hoverinfo="skip",
    ))

    # Forecast line (dashed)
    fig.add_trace(go.Scatter(
        x=fcst2.index,
        y=fcst2["yhat"].values,
        mode="lines",
        name="Forecast",
        line=dict(color=col_fcst, width=2.6, dash="dash"),
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        showlegend=True,
    )

    # Auto-zoom: keep intraday forecasts readable
    try:
        if len(hist2) > 10:
            n_hist = min(len(hist2), 400)
            x0 = hist2.index[-n_hist]
            x1 = fcst2.index[-1]
            fig.update_xaxes(range=[x0, x1])
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)


# ============================
# Strategy output container
# ============================

@dataclass
class StrategyOutput:
    """Standard container used for plotting and metrics."""
    name: str
    equity: pd.Series
    returns: pd.Series


def find_output(outputs: List[StrategyOutput], startswith: str) -> Optional[StrategyOutput]:
    """Utility to find a strategy output by prefix."""
    for o in outputs:
        if o.name.startswith(startswith):
            return o
    return None


# ============================
# Warm-up / slicing / rebasing
# ============================

def _bars_per_day(interval: str) -> float:
    """
    Rough mapping used to translate "bars needed" into calendar days for warm-up fetching.
    - 1d: 1 bar/day
    - 60m: approximate US session 6.5 hours => ~6.5 bars/day
    """
    return 1.0 if interval == "1d" else 6.5


def compute_warmup_days(interval: str, bars_needed: int) -> int:
    """
    Fetch extra history BEFORE the user's start date so rolling indicators have enough observations.
    """
    if bars_needed <= 0:
        return 0
    bpd = _bars_per_day(interval)
    days = int(np.ceil(bars_needed / bpd))
    if interval == "1d":
        return max(10, 2 * days)
    return max(10, 2 * days + 5)


def user_window_bounds(start: date, end: date) -> Tuple[datetime, datetime]:
    """Convert date inputs into inclusive-start / exclusive-end datetimes."""
    start_dt = datetime.combine(start, time.min)
    end_dt_excl = datetime.combine(end, time.min) + timedelta(days=1)
    return start_dt, end_dt_excl


def slice_series_to_user_window(s: pd.Series, start_dt: datetime, end_dt_excl: datetime) -> pd.Series:
    """Slice a time series to [start_dt, end_dt_excl)."""
    if s is None or s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    return s.loc[(s.index >= start_dt) & (s.index < end_dt_excl)]


def rebase_equity(eq: pd.Series) -> pd.Series:
    """Rebase equity to 1.0 at the start of the user window."""
    if eq is None or eq.empty:
        return eq
    first = float(eq.iloc[0])
    if not np.isfinite(first) or first == 0.0:
        return eq
    return eq / first


# ============================
# Strategies (long-only)
# ============================

def strat_buy_hold(price: pd.Series) -> StrategyOutput:
    """Benchmark: always long."""
    r = to_returns(price)
    eq = equity_from_returns(r)
    return StrategyOutput("Buy & Hold", eq, r)


def strat_sma_crossover_full(
    price_full: pd.Series,
    short_window: int,
    long_window: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """SMA crossover with next-bar execution (shifted position)."""
    ma_s = price_full.rolling(short_window).mean()
    ma_l = price_full.rolling(long_window).mean()
    position = (ma_s > ma_l).astype(float).shift(1).fillna(0.0)
    return position, ma_s, ma_l


def strat_sma_crossover(
    price_full: pd.Series,
    start_dt: datetime,
    end_dt_excl: datetime,
    short_window: int,
    long_window: int,
) -> StrategyOutput:
    pos_full, _, _ = strat_sma_crossover_full(price_full, short_window, long_window)
    r_full = to_returns(price_full)
    strat_r_full = r_full * pos_full
    eq_full = equity_from_returns(strat_r_full)

    strat_r = slice_series_to_user_window(strat_r_full, start_dt, end_dt_excl)
    eq = rebase_equity(slice_series_to_user_window(eq_full, start_dt, end_dt_excl))
    return StrategyOutput(f"SMA Crossover ({short_window},{long_window})", eq, strat_r)


def _rsi(price: pd.Series, window: int) -> pd.Series:
    """Simple RSI (rolling mean of gains/losses), intentionally simple."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def rsi_position_state_machine(rsi: pd.Series, low: float, high: float) -> pd.Series:
    """RSI mean reversion (long-only) with next-bar execution."""
    pos = pd.Series(0.0, index=rsi.index)
    in_pos = False

    for i in range(len(rsi)):
        rv = float(rsi.iloc[i])
        if (not in_pos) and (rv < low):
            in_pos = True
        elif in_pos and (rv > high):
            in_pos = False
        pos.iloc[i] = 1.0 if in_pos else 0.0

    return pos.shift(1).fillna(0.0)


def strat_rsi_mean_reversion(price: pd.Series, rsi_window: int, low: int, high: int) -> StrategyOutput:
    rsi = _rsi(price, rsi_window)
    pos = rsi_position_state_machine(rsi, low=float(low), high=float(high))
    r = to_returns(price)
    strat_r = r * pos
    eq = equity_from_returns(strat_r)
    return StrategyOutput(f"RSI Mean Reversion (w={rsi_window}, {low}/{high})", eq, strat_r)


def strat_bollinger_full(
    price_full: pd.Series,
    window: int,
    n_std: float,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger bands + mean reversion (long-only) with next-bar execution."""
    ma = price_full.rolling(window).mean()
    sd = price_full.rolling(window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd

    pos = pd.Series(0.0, index=price_full.index)
    pos[price_full < lower] = 1.0
    pos[price_full > ma] = 0.0
    pos = pos.shift(1).fillna(0.0)
    return pos, ma, upper, lower


def strat_bollinger_mean_reversion(
    price_full: pd.Series,
    start_dt: datetime,
    end_dt_excl: datetime,
    window: int,
    n_std: float,
) -> StrategyOutput:
    pos_full, _, _, _ = strat_bollinger_full(price_full, window, n_std)
    r_full = to_returns(price_full)
    strat_r_full = r_full * pos_full
    eq_full = equity_from_returns(strat_r_full)

    strat_r = slice_series_to_user_window(strat_r_full, start_dt, end_dt_excl)
    eq = rebase_equity(slice_series_to_user_window(eq_full, start_dt, end_dt_excl))
    return StrategyOutput(f"Bollinger MR (w={window}, sigma={n_std:.1f})", eq, strat_r)


def strat_donchian_full(price_full: pd.Series, window: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian breakout (long-only) with next-bar execution."""
    hi = price_full.rolling(window).max().shift(1)
    lo = price_full.rolling(window).min().shift(1)

    pos = pd.Series(0.0, index=price_full.index)
    in_pos = False

    for i in range(len(price_full)):
        h = hi.iloc[i]
        l = lo.iloc[i]
        if pd.isna(h) or pd.isna(l):
            pos.iloc[i] = 0.0
            continue

        c = float(price_full.iloc[i])
        if (not in_pos) and (c > float(h)):
            in_pos = True
        elif in_pos and (c < float(l)):
            in_pos = False

        pos.iloc[i] = 1.0 if in_pos else 0.0

    pos = pos.shift(1).fillna(0.0)
    return pos, hi, lo


def strat_donchian_breakout(
    price_full: pd.Series,
    start_dt: datetime,
    end_dt_excl: datetime,
    window: int,
) -> StrategyOutput:
    pos_full, _, _ = strat_donchian_full(price_full, window)
    r_full = to_returns(price_full)
    strat_r_full = r_full * pos_full
    eq_full = equity_from_returns(strat_r_full)

    strat_r = slice_series_to_user_window(strat_r_full, start_dt, end_dt_excl)
    eq = rebase_equity(slice_series_to_user_window(eq_full, start_dt, end_dt_excl))
    return StrategyOutput(f"Donchian Breakout (w={window})", eq, strat_r)


# ============================
# Data loader (yfinance) + "live" last price
# ============================

def _drop_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Make DatetimeIndex timezone-naive to avoid tz-aware comparisons."""
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_ohlcv(ticker: str, start: date, end: date, interval: str, cache_buster: int) -> pd.DataFrame:
    """Download OHLCV bars from Yahoo Finance via yfinance."""
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end, time.min) + timedelta(days=1)

    df = yf.download(
        ticker,
        start=start_dt,
        end=end_dt,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return _drop_tz(df)


@st.cache_data(ttl=300, show_spinner=False)
def load_live_last_price(ticker: str, cache_buster: int) -> float:
    """Best-effort near-real-time last price (display only)."""
    try:
        t = yf.Ticker(ticker)

        # 1-minute bars
        try:
            h1 = _drop_tz(t.history(period="1d", interval="1m"))
            if h1 is not None and not h1.empty and "Close" in h1.columns:
                v = float(h1["Close"].dropna().iloc[-1])
                if np.isfinite(v):
                    return v
        except Exception:
            pass

        # 5-minute bars
        try:
            h5 = _drop_tz(t.history(period="1d", interval="5m"))
            if h5 is not None and not h5.empty and "Close" in h5.columns:
                v = float(h5["Close"].dropna().iloc[-1])
                if np.isfinite(v):
                    return v
        except Exception:
            pass

        # fast_info
        try:
            fi = getattr(t, "fast_info", None)
            if fi and "last_price" in fi and fi["last_price"] is not None:
                v = float(fi["last_price"])
                if np.isfinite(v):
                    return v
        except Exception:
            pass

    except Exception:
        return float("nan")

    return float("nan")


def _select_price_series(df: pd.DataFrame, interval: str) -> pd.Series:
    """Pick Close (intraday) or Adj Close if available (daily)."""
    if interval != "1d":
        if "Close" not in df.columns:
            raise ValueError("No 'Close' column found in OHLCV data.")
        s = df["Close"]
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("No 'Adj Close' or 'Close' column found in OHLCV data.")

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def append_live_point_for_display(price_window: pd.Series, live_price: float) -> pd.Series:
    """Append a synthetic (now, live_price) point to the display chart only."""
    if price_window is None or price_window.empty or not np.isfinite(live_price):
        return price_window

    now = pd.Timestamp.now().to_pydatetime().replace(tzinfo=None)
    last_ts = price_window.index[-1]
    last_val = float(price_window.iloc[-1])

    if np.isfinite(last_val) and abs(live_price - last_val) / max(abs(last_val), 1e-9) < 1e-6:
        return price_window

    if isinstance(last_ts, pd.Timestamp):
        last_ts = last_ts.to_pydatetime()
    if now <= last_ts:
        return price_window

    s2 = price_window.copy()
    s2.loc[now] = live_price
    return s2.sort_index()


# ============================
# Last price indicator (metric-style, like before)
# ============================

def render_last_price_metric(last_price: float, last_bar_close: float) -> None:
    """
    Display last price with Streamlit metric delta (no dot).
    Delta is computed vs the last available bar close in the selected window.
    """
    if not np.isfinite(last_price):
        st.metric("Last price", "N/A")
        return

    delta = float("nan")
    pct = float("nan")
    if np.isfinite(last_bar_close) and last_bar_close != 0:
        delta = float(last_price - last_bar_close)
        pct = float(delta / last_bar_close)

    if np.isfinite(delta) and np.isfinite(pct):
        st.metric("Last price", f"{last_price:.2f}", f"{delta:+.2f} ({pct:+.2%})")
    elif np.isfinite(delta):
        st.metric("Last price", f"{last_price:.2f}", f"{delta:+.2f}")
    else:
        st.metric("Last price", f"{last_price:.2f}")


# ============================
# Forecasting (ML tab)
# ============================

def _clean_price_for_ml(price: pd.Series) -> pd.Series:
    """Clean price series for ML: numeric, sorted, unique timestamps."""
    y = pd.to_numeric(price, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if y.empty:
        return y
    y = y.sort_index()
    y = y[~y.index.duplicated(keep="last")]
    return y


def _infer_future_freq(price: pd.Series, interval: str) -> str:
    """
    Choose a reasonable future frequency string for pd.date_range.
    - Daily: business days ("B")
    - Intraday: infer from median delta and return Pandas aliases ("H", "T", etc.)
    """
    if interval == "1d":
        return "B"

    idx = pd.DatetimeIndex(price.index)
    if len(idx) < 3:
        return "H"

    deltas = (idx[1:] - idx[:-1]).to_series().dropna()
    if deltas.empty:
        return "H"

    med = deltas.tail(200).median()

    # Snap to 60 minutes if close
    if abs(med - pd.Timedelta(hours=1)) <= pd.Timedelta(minutes=5):
        return "H"

    # Otherwise: infer minutes and use Pandas "T" alias (not "min")
    mins = int(max(1, round(med.total_seconds() / 60.0)))
    return f"{mins}T"


def _z_from_alpha(alpha: float) -> float:
    """Minimal alpha->z mapping (kept intentionally simple for the assignment)."""
    z = 1.96
    if np.isfinite(alpha) and 0 < alpha < 1:
        if abs(alpha - 0.10) < 1e-9:
            z = 1.645
        elif abs(alpha - 0.05) < 1e-9:
            z = 1.96
        elif abs(alpha - 0.01) < 1e-9:
            z = 2.576
    return z


def _market_signature(idx: pd.DatetimeIndex) -> set:
    """
    Build the set of (weekday, hour, minute) observed in history.
    This lets us filter Prophet's future timestamps to match market hours structure.
    """
    sig = set()
    for ts in idx:
        sig.add((ts.weekday(), ts.hour, ts.minute))
    return sig


def _future_market_index_from_history(
    last_ts: pd.Timestamp,
    freq: str,
    horizon: int,
    allowed_sig: set,
) -> pd.DatetimeIndex:
    """
    Generate a future DatetimeIndex of length=horizon, stepping with `freq`,
    and keeping only timestamps whose (weekday, hour, minute) appeared in history.
    """
    if horizon <= 0:
        return pd.DatetimeIndex([])

    try:
        step = pd.tseries.frequencies.to_offset(freq)
    except Exception:
        step = pd.tseries.frequencies.to_offset("H")

    out: List[pd.Timestamp] = []
    cur = pd.Timestamp(last_ts)

    # Safety to avoid infinite loops (e.g. if allowed_sig is empty)
    max_iter = int(max(10_000, horizon * 50))
    it = 0

    while len(out) < horizon and it < max_iter:
        it += 1
        cur = cur + step
        if (cur.weekday(), cur.hour, cur.minute) in allowed_sig:
            out.append(cur)

    return pd.DatetimeIndex(out)


@st.cache_data(ttl=300, show_spinner=False)
def forecast_linear_regression(
    price: pd.Series,
    interval: str,
    horizon: int,
    alpha: float,
    cache_buster: int,
) -> pd.DataFrame:
    """
    Baseline model: linear trend fitted on a recent window, anchored to last observed price.

    Confidence interval:
    - Uses a widening prediction interval ("cone") based on standard regression prediction
      uncertainty, which grows with distance from the fitted sample.
    """
    if price is None or price.empty or horizon <= 0:
        return pd.DataFrame()

    y = _clean_price_for_ml(price)
    if len(y) < 20:
        return pd.DataFrame()

    # Recent window for a realistic slope, especially intraday
    lookback = int(min(len(y), max(80, 10 * horizon)))
    y_tail = y.iloc[-lookback:].astype(float)

    n = len(y_tail)
    x = np.arange(n, dtype=float)

    # Fit slope/intercept
    a, b = np.polyfit(x, y_tail.values, deg=1)

    # Anchor to last observed price (eliminate jump)
    last_y = float(y_tail.iloc[-1])
    x_last = float(n - 1)
    yhat_last = float(a * x_last + b)
    b = b + (last_y - yhat_last)

    # Residual std
    y_fit = a * x + b
    resid = y_tail.values - y_fit
    sigma = float(np.std(resid, ddof=1)) if n > 2 else float(np.std(resid, ddof=0))

    # Regression terms for prediction interval
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))

    z = _z_from_alpha(alpha)

    # Build forecast index INCLUDING anchor row at last timestamp
    freq = _infer_future_freq(y, interval)
    last_ts = pd.Timestamp(y.index[-1])
    idx_fcst = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)

    # Predict for x_future = x_last + k, k=0..horizon
    k = np.arange(0, horizon + 1, dtype=float)
    x_future = x_last + k
    yhat = a * x_future + b  # anchored at k=0

    # Prediction standard error (cone widening with horizon)
    if np.isfinite(sigma) and sigma > 0:
        if sxx > 0:
            se_pred = sigma * np.sqrt(1.0 + (1.0 / n) + ((x_future - x_bar) ** 2) / sxx)
        else:
            se_pred = sigma * np.sqrt(1.0 + (1.0 / n))
    else:
        se_pred = np.zeros_like(yhat)

    band = z * se_pred
    yhat_lower = yhat - band
    yhat_upper = yhat + band

    df_fcst = pd.DataFrame(
        {
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
        },
        index=idx_fcst,
    )

    # Hard-set anchor to exact last price (visual + numeric stability)
    df_fcst.iloc[0, df_fcst.columns.get_loc("yhat")] = last_y

    return df_fcst


@st.cache_data(ttl=300, show_spinner=False)
def forecast_prophet(
    price: pd.Series,
    interval: str,
    horizon: int,
    interval_width: float,
    cache_buster: int,
) -> pd.DataFrame:
    """
    Prophet forecast (optional).

    Robustness improvements:
    - Intraday: future timestamps are filtered to match market-hour patterns observed in history.
    - Forecast is anchored to last observed price via an offset (no jump).
    - Output includes an anchor row at the last timestamp.
    """
    if price is None or price.empty or horizon <= 0:
        return pd.DataFrame()

    try:
        try:
            from prophet import Prophet
        except Exception:
            from fbprophet import Prophet  # type: ignore
    except Exception:
        return pd.DataFrame()

    y = _clean_price_for_ml(price)
    if len(y) < 50:
        return pd.DataFrame()

    last_ts = pd.Timestamp(y.index[-1])
    last_y = float(y.iloc[-1])

    df = pd.DataFrame({"ds": pd.to_datetime(y.index), "y": y.values})

    # Model settings (kept simple + stable)
    if interval == "1d":
        m = Prophet(
            interval_width=float(interval_width),
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        freq = "B"
    else:
        m = Prophet(
            interval_width=float(interval_width),
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=5.0,
        )
        # Optional intra-day seasonality with low Fourier order
        try:
            m.add_seasonality(name="intra_day", period=1.0, fourier_order=3)
        except Exception:
            pass
        freq = _infer_future_freq(y, interval)

    m.fit(df)

    # Compute model-implied yhat at last timestamp (for anchoring)
    pred_last = m.predict(pd.DataFrame({"ds": [last_ts]}))
    yhat_last_model = float(pred_last["yhat"].iloc[0])
    offset = last_y - yhat_last_model

    # Build future timestamps
    if interval == "1d":
        future = m.make_future_dataframe(periods=int(horizon), freq=freq, include_history=False)
        fcst = m.predict(future)
        out = fcst.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].copy()
        out["yhat"] += offset
        out["yhat_lower"] += offset
        out["yhat_upper"] += offset

        # Anchor row uses Prophet's own uncertainty at last point (shifted)
        anchor = pd.DataFrame(
            {
                "yhat": [last_y],
                "yhat_lower": [float(pred_last["yhat_lower"].iloc[0]) + offset],
                "yhat_upper": [float(pred_last["yhat_upper"].iloc[0]) + offset],
            },
            index=pd.DatetimeIndex([last_ts]),
        )

        out = pd.concat([anchor, out], axis=0).sort_index()
        return out

    # Intraday: filter future timestamps to match observed (weekday,hour,minute)
    allowed = _market_signature(pd.DatetimeIndex(y.index))
    future_idx = _future_market_index_from_history(
        last_ts=last_ts, freq=freq, horizon=int(horizon), allowed_sig=allowed
    )
    if len(future_idx) == 0:
        return pd.DataFrame()

    future = pd.DataFrame({"ds": future_idx})
    fcst = m.predict(future)
    out = fcst.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].copy()
    out["yhat"] += offset
    out["yhat_lower"] += offset
    out["yhat_upper"] += offset

    # Anchor row with uncertainty at last point (shifted)
    anchor = pd.DataFrame(
        {
            "yhat": [last_y],
            "yhat_lower": [float(pred_last["yhat_lower"].iloc[0]) + offset],
            "yhat_upper": [float(pred_last["yhat_upper"].iloc[0]) + offset],
        },
        index=pd.DatetimeIndex([last_ts]),
    )

    out = pd.concat([anchor, out], axis=0).sort_index()
    return out


# ============================
# Streamlit page
# ============================

def run_single_asset():
    """Main Streamlit entrypoint."""
    st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

    st.title("Single Asset Analysis (Quant A)")
    st.caption("Yahoo Finance data, backtests, metrics, and a forecasting tab.")

    cache_buster = int(pytime.time() // 300)

    # ----------------------------
    # Sidebar controls
    # ----------------------------
    st.sidebar.header("Settings")
    st.sidebar.caption(
        "Ticker tips:\n"
        "- FR equities: ENGI.PA, TTE.PA, BNP.PA\n"
        "- FX: EURUSD=X\n"
        "- Crypto: BTC-USD\n"
    )

    # Default ticker changed to BNP.PA
    ticker = st.sidebar.text_input("Ticker", value="BNP.PA").strip().upper()

    interval_ui = st.sidebar.selectbox("Interval", ["1D", "1H"], index=0)
    interval = "1d" if interval_ui == "1D" else "60m"

    periods = TRADING_DAYS if interval == "1d" else int(252 * 6.5)

    today = date.today()
    default_start = today - timedelta(days=365)
    max_end = today + timedelta(days=1)

    start = st.sidebar.date_input("Start date", value=default_start, max_value=max_end)
    end = st.sidebar.date_input("End date", value=today, max_value=max_end)

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
    short_window = st.sidebar.slider("Short SMA (bars)", 5, 50, 20, 1)
    long_window = st.sidebar.slider("Long SMA (bars)", 20, 200, 50, 5)

    st.sidebar.subheader("RSI params")
    rsi_window = st.sidebar.slider("RSI window (bars)", 5, 50, 14, 1)
    rsi_low = st.sidebar.slider("Oversold threshold", 5, 45, 30, 1)
    rsi_high = st.sidebar.slider("Overbought threshold", 55, 95, 70, 1)

    st.sidebar.subheader("Bollinger params")
    bb_window = st.sidebar.slider("BB window (bars)", 10, 100, 20, 1)
    bb_nstd = st.sidebar.slider("BB std (σ)", 1.0, 3.0, 1.5, 0.1)

    st.sidebar.subheader("Donchian params")
    don_window = st.sidebar.slider("Donchian window (bars)", 10, 200, 50, 5)

    # ----------------------------
    # Validations
    # ----------------------------
    if not ticker:
        st.error("Please enter a valid ticker (e.g., BNP.PA, AAPL, ENGI.PA, EURUSD=X).")
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

    # ----------------------------
    # Warm-up
    # ----------------------------
    bars_needed = 0
    if "SMA Crossover" in selected:
        bars_needed = max(bars_needed, long_window + 5)
    if "Donchian Breakout" in selected:
        bars_needed = max(bars_needed, don_window + 5)
    if "Bollinger Mean Reversion" in selected:
        bars_needed = max(bars_needed, bb_window + 5)

    warmup_days = compute_warmup_days(interval=interval, bars_needed=bars_needed)
    start_ext = start - timedelta(days=warmup_days)
    start_dt, end_dt_excl = user_window_bounds(start, end)

    # ----------------------------
    # Data load
    # ----------------------------
    with st.spinner(f"Downloading data for {ticker} ({interval_ui})..."):
        df_full = load_ohlcv(ticker, start_ext, end, interval=interval, cache_buster=cache_buster)

    if df_full.empty:
        st.warning(
            "No data found.\n\n"
            "- FR equities: add .PA\n"
            "- FX: use =X\n"
            "- Intraday depends on Yahoo Finance\n"
        )
        return

    live_last = load_live_last_price(ticker, cache_buster=cache_buster)

    try:
        price_full = _select_price_series(df_full, interval=interval)
    except ValueError as e:
        st.error(str(e))
        return

    price = slice_series_to_user_window(price_full, start_dt, end_dt_excl)
    if price.empty:
        st.warning("No data inside the selected window. Try adjusting dates.")
        return

    price_display = append_live_point_for_display(price, live_last)

    # ----------------------------
    # Compute strategies
    # ----------------------------
    outputs: List[StrategyOutput] = []
    if "Buy & Hold" in selected:
        outputs.append(strat_buy_hold(price))
    if "RSI Mean Reversion" in selected:
        outputs.append(strat_rsi_mean_reversion(price, rsi_window, rsi_low, rsi_high))
    if "SMA Crossover" in selected:
        outputs.append(strat_sma_crossover(price_full, start_dt, end_dt_excl, short_window, long_window))
    if "Bollinger Mean Reversion" in selected:
        outputs.append(strat_bollinger_mean_reversion(price_full, start_dt, end_dt_excl, bb_window, bb_nstd))
    if "Donchian Breakout" in selected:
        outputs.append(strat_donchian_breakout(price_full, start_dt, end_dt_excl, don_window))

    equity_df = pd.concat([o.equity.rename(o.name) for o in outputs], axis=1)
    equity_df = equity_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    price_norm = (price / float(price.iloc[0])).rename("Price (normalized)")
    combined_df = pd.concat([price_norm, equity_df], axis=1)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    metrics_rows = {o.name: compute_metrics(o.equity, o.returns, periods=int(periods)) for o in outputs}
    metrics_df = pd.DataFrame(metrics_rows).T

    tab_overview, tab_strat, tab_ml, tab_data = st.tabs(["Overview", "Strategies", "ML Forecast", "Data"])

    # ----------------------------
    # OVERVIEW
    # ----------------------------
    with tab_overview:
        last_bar_close = float(price.iloc[-1])
        last_price = float(live_last) if np.isfinite(live_last) else last_bar_close

        # Metric-style indicator (no dot)
        render_last_price_metric(last_price=last_price, last_bar_close=last_bar_close)

        st.subheader("Raw price")
        plot_series_plotly(f"{ticker} price ({interval_ui})", price_display.rename("Price"), y_label="Price")

        st.subheader("Normalized price vs strategies")
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

    # ----------------------------
    # STRATEGIES
    # ----------------------------
    with tab_strat:
        st.subheader("Equity curves")
        plot_df_plotly("Equity curves", equity_df, y_label="Equity")

        st.divider()
        st.subheader("Per-strategy diagnostics")

        if "SMA Crossover" in selected:
            with st.expander("SMA Crossover diagnostics", expanded=True):
                pos_full, ma_s_full, ma_l_full = strat_sma_crossover_full(price_full, short_window, long_window)

                ma_s = slice_series_to_user_window(ma_s_full, start_dt, end_dt_excl).rename(f"SMA {short_window}")
                ma_l = slice_series_to_user_window(ma_l_full, start_dt, end_dt_excl).rename(f"SMA {long_window}")
                pos = slice_series_to_user_window(pos_full, start_dt, end_dt_excl)

                plot_df_plotly("Price + SMAs", pd.concat([price.rename("Price"), ma_s, ma_l], axis=1), y_label="Price")
                plot_df_plotly("Position (1=long; shifted)", pd.DataFrame({"Position": pos}), y_label="Position")

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

                pos = rsi_position_state_machine(rsi, low=float(rsi_low), high=float(rsi_high))
                plot_df_plotly("Position (1=long; shifted)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "RSI Mean Reversion")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

        if "Bollinger Mean Reversion" in selected:
            with st.expander("Bollinger Mean Reversion diagnostics", expanded=False):
                pos_full, ma_full, upper_full, lower_full = strat_bollinger_full(price_full, bb_window, bb_nstd)

                ma = slice_series_to_user_window(ma_full, start_dt, end_dt_excl).rename("Middle")
                upper = slice_series_to_user_window(upper_full, start_dt, end_dt_excl).rename("Upper")
                lower = slice_series_to_user_window(lower_full, start_dt, end_dt_excl).rename("Lower")
                pos = slice_series_to_user_window(pos_full, start_dt, end_dt_excl)

                plot_df_plotly(
                    "Price + Bollinger Bands",
                    pd.concat([price.rename("Price"), ma, upper, lower], axis=1),
                    y_label="Price",
                )
                plot_df_plotly("Position (1=long; shifted)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "Bollinger MR")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

        if "Donchian Breakout" in selected:
            with st.expander("Donchian Breakout diagnostics", expanded=False):
                pos_full, hi_full, lo_full = strat_donchian_full(price_full, don_window)
                hi = slice_series_to_user_window(hi_full, start_dt, end_dt_excl).rename("Channel High")
                lo = slice_series_to_user_window(lo_full, start_dt, end_dt_excl).rename("Channel Low")
                pos = slice_series_to_user_window(pos_full, start_dt, end_dt_excl)

                plot_df_plotly(
                    "Price + Donchian Channel",
                    pd.concat([price.rename("Price"), hi, lo], axis=1),
                    y_label="Price",
                )
                plot_df_plotly("Position (1=long; shifted)", pd.DataFrame({"Position": pos}), y_label="Position")

                o = find_output(outputs, "Donchian Breakout")
                if o is not None:
                    plot_series_plotly("Equity curve", o.equity, y_label="Equity")

    # ----------------------------
    # ML FORECAST
    # ----------------------------
    with tab_ml:
        # Removed "Optional Bonus" header and the long caption as requested
        st.subheader("Forecasting")

        c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
        with c1:
            model = st.selectbox("Model", ["None", "Linear Trend (baseline)", "Prophet (optional)"], index=1)
        with c2:
            default_h = 5 if interval == "1d" else 24
            horizon = st.number_input("Horizon (bars)", min_value=1, max_value=500, value=int(default_h), step=1)
        with c3:
            ci = st.selectbox("Confidence level", ["90%", "95%", "99%"], index=1)
            if ci == "90%":
                alpha = 0.10
                interval_width = 0.90
            elif ci == "99%":
                alpha = 0.01
                interval_width = 0.99
            else:
                alpha = 0.05
                interval_width = 0.95

        # Train on full window, plot a tail so the forecast is readable (especially for 1H)
        hist_train = price.copy()
        hist_plot = price.tail(400) if interval != "1d" else price.tail(200)

        if model == "None":
            st.info("Select a model to run a forecast.")
        elif model.startswith("Linear Trend"):
            fcst = forecast_linear_regression(
                price=hist_train,
                interval=interval,
                horizon=int(horizon),
                alpha=float(alpha),
                cache_buster=cache_buster,
            )
            if fcst.empty:
                st.warning("Not enough data to run the forecast (try a longer date range).")
            else:
                plot_forecast_plotly(
                    title=f"{ticker} — Linear Trend forecast ({interval_ui})",
                    hist=hist_plot,
                    fcst=fcst,
                    y_label="Price",
                )
        else:
            fcst = forecast_prophet(
                price=hist_train,
                interval=interval,
                horizon=int(horizon),
                interval_width=float(interval_width),
                cache_buster=cache_buster,
            )
            if fcst.empty:
                st.warning("Prophet unavailable (not installed or not enough data).")
                st.info("Install: `pip install prophet` (in your venv/terminal).")
            else:
                plot_forecast_plotly(
                    title=f"{ticker} — Prophet forecast ({interval_ui})",
                    hist=hist_plot,
                    fcst=fcst,
                    y_label="Price",
                )

    # ----------------------------
    # DATA
    # ----------------------------
    with tab_data:
        st.subheader("OHLCV data (tail)")
        st.dataframe(df_full.tail(50), use_container_width=True)
