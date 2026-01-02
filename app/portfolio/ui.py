# app/portfolio/ui.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.common.data_fetcher import get_price_series


# -------------------------
# Helpers: metrics & logic
# -------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna(how="all")
    return rets.dropna(axis=1, how="all")


def annualize_return(daily_mean: pd.Series, periods: int = 252) -> pd.Series:
    return daily_mean * periods


def annualize_vol(daily_std: pd.Series, periods: int = 252) -> pd.Series:
    return daily_std * np.sqrt(periods)


def sharpe_ratio(ann_ret: float, ann_vol: float, rf: float = 0.0) -> float:
    if ann_vol <= 0:
        return np.nan
    return (ann_ret - rf) / ann_vol


def max_drawdown(series: pd.Series) -> float:
    # series: portfolio value
    peak = series.cummax()
    dd = (series / peak) - 1.0
    return float(dd.min())


def normalize_weights(symbols: list[str], weights: list[float]) -> pd.Series:
    w = pd.Series(weights, index=symbols, dtype=float)
    if (w < 0).any():
        raise ValueError("Negative weights prohibited")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("The sum of the weights must be > 0.")
    return w / s


def rebalance_portfolio(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    rebalance: str = "None",
    initial_value: float = 100.0,
) -> pd.Series:
    """
    Returns portfolio value series.
    rebalance: 'None', 'W', 'M', 'Q'
    """
    prices = prices.dropna(how="all").ffill().dropna()
    rets = compute_returns(prices)

    if rebalance == "None":
        # Buy-and-hold with initial weights
        port_rets = (rets * target_weights).sum(axis=1)
        return (1.0 + port_rets).cumprod() * initial_value

    # With periodic rebalancing: reset weights at period boundaries
    rule = {"W": "W-FRI", "M": "M", "Q": "Q"}.get(rebalance, None)
    if rule is None:
        raise ValueError("Rebalancing unknown.")

    # Create rebalance dates based on returns index
    reb_dates = rets.resample(rule).last().index
    reb_dates = set(reb_dates)

    w = target_weights.copy()
    value = initial_value
    values = []

    for dt, row in rets.iterrows():
        # Apply daily return with current weights
        port_ret = float((row * w).sum())
        value *= (1.0 + port_ret)
        values.append(value)

        # Rebalance at end of period (when dt matches a rebalance date)
        if dt in reb_dates:
            w = target_weights.copy()

    return pd.Series(values, index=rets.index, name="Portfolio Value")


def portfolio_summary(prices: pd.DataFrame, w: pd.Series, rf: float = 0.0) -> dict:
    rets = compute_returns(prices)
    port_rets = (rets * w).sum(axis=1)

    mean_daily = float(port_rets.mean())
    vol_daily = float(port_rets.std())

    ann_ret = mean_daily * 252
    ann_vol = vol_daily * np.sqrt(252)
    sr = sharpe_ratio(ann_ret, ann_vol, rf=rf)

    nav = (1.0 + port_rets).cumprod() * 100.0
    mdd = max_drawdown(nav)

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sr,
        "max_drawdown": mdd,
        "nav": nav,
        "port_rets": port_rets,
        "rets": rets,
    }


def diversification_indicators(rets: pd.DataFrame, w: pd.Series) -> dict:
    corr = rets.corr()
    # average pairwise correlation (upper triangle excluding diagonal)
    vals = corr.values
    n = vals.shape[0]
    if n <= 1:
        avg_corr = np.nan
    else:
        avg_corr = float(vals[np.triu_indices(n, k=1)].mean())

    # Effective number of assets (simple concentration proxy)
    en = float(1.0 / np.sum(np.square(w.values)))

    return {
        "avg_pairwise_corr": avg_corr,
        "effective_n": en,
        "corr": corr,
    }


def plot_main(prices: pd.DataFrame, portfolio_value: pd.Series) -> go.Figure:
    fig = go.Figure()

    # Normalize asset prices to 100 (visual comparison)
    base = prices.iloc[0]
    norm = (prices / base) * 100.0

    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=f"{col}"))

    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode="lines",
        name="Portfolio Value",
        line=dict(width=3),
    ))

    fig.update_layout(
        title="Asset Prices (Normalized) & Cumulative Portfolio Value:",
        xaxis_title="Date",
        yaxis_title="Base 100",
        legend_title="Series",
        height=600,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def plot_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1, zmax=1,
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(
        title="Correlation Matrix (Returns)",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def markowitz_monte_carlo(rets: pd.DataFrame, rf: float = 0.0, n_portfolios: int = 6000, seed: int = 42):
    """
    Random long-only portfolios. Returns dataframe with weights & metrics.
    """
    rng = np.random.default_rng(seed)
    assets = list(rets.columns)
    mu = rets.mean() * 252
    cov = rets.cov() * 252

    records = []
    for _ in range(n_portfolios):
        w = rng.random(len(assets))
        w = w / w.sum()

        ann_ret = float(np.dot(w, mu.values))
        ann_vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
        sr = sharpe_ratio(ann_ret, ann_vol, rf=rf)
        records.append([ann_ret, ann_vol, sr, *w])

    cols = ["ann_return", "ann_vol", "sharpe", *assets]
    return pd.DataFrame(records, columns=cols)


def plot_frontier(df_mc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_mc["ann_vol"],
        y=df_mc["ann_return"],
        mode="markers",
        name="Simulated Portfolios",
        marker=dict(size=5, opacity=0.6),
        text=[f"Sharpe={s:.2f}" for s in df_mc["sharpe"]],
        hovertemplate="Vol=%{x:.2%}<br>Return=%{y:.2%}<br>%{text}<extra></extra>"
    ))
    fig.update_layout(
        title="Markowitz (Monte-Carlo) : Return vs Volatility",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


# -------------------------
# UI: Portfolio module
# -------------------------
def run_portfolio():
    st.header("Portfolio Management & Analysis (Minimum 3 Assets) - Quant B")

    # Auto-refresh every 5 minutes (works well with cache ttl=300 too)
    st.caption("Auto-Refresh Every 5 Minutes (Cache Data + UI Refresh).")
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5 * 60 * 1000, key="portfolio_autorefresh")
    except Exception:
        # If streamlit-autorefresh not installed, no hard fail
        pass

    # --- Inputs (shared across tabs)
    with st.sidebar:
        st.subheader("Portfolio Parameters")
        # --- Tickers UI (champs dynamiques) ---
        MAX_TICKERS = 30
        DEFAULT_TICKERS = ["V", "MC.PA", "2330.TW"]  # Visa, LVMH, TSMC

        # Init session state (1 fois)
        if "ticker_inputs" not in st.session_state:
        # 3 tickers par défaut + 1 case vide
            st.session_state.ticker_inputs = DEFAULT_TICKERS + [""]

            st.sidebar.caption("Add tickers: a new field appears as soon as you type in the last one.")

        # Affiche les champs existants
        old_len = len(st.session_state.ticker_inputs)
        for i in range(len(st.session_state.ticker_inputs)):
            st.session_state.ticker_inputs[i] = st.sidebar.text_input(
                f"Ticker {i+1}",
                value=st.session_state.ticker_inputs[i],
                key=f"ticker_{i}",
                placeholder="Ex: AIR.PA"
            ).strip().upper()

        # Delete empty tickers 
        filled = [t for t in st.session_state.ticker_inputs if t != ""]
        st.session_state.ticker_inputs = filled + [""]  # always 1 room for an aditional ticker
        # Si le compactage a créé une nouvelle case (ex: on a rempli la dernière), on rerun pour l'afficher
        if len(st.session_state.ticker_inputs) != old_len:
            st.rerun()



        # Construire la liste finale des tickers (en supprimant les vides)
        symbols = [t for t in st.session_state.ticker_inputs if t != ""]


        if len(symbols) < 3:
            st.sidebar.warning("⚠️ Please provide at least 3 tickers for the Quant B module.")

        start = st.date_input("Start Date", value=pd.to_datetime("2020-01-01").date())
        end = st.date_input("End Date", value=pd.to_datetime("2025-01-01").date())

        interval = st.selectbox("Interval", ["1d", "1h"], index=0)
        rf_pct = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.125, step=0.001, format="%.3f",)
        rf = rf_pct / 100.0


        weight_mode = st.radio("Weighting", ["Equal Weight", "Custom Weights"], horizontal=True)

        rebalance_label = st.selectbox("Rebalancing", ["None", "Weekly", "Monthly", "Quarterly"], index=2)
        reb_map = {"None": "None", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
        reb = reb_map[rebalance_label]

    # Weights UI
    if weight_mode == "Equal Weight":
        weights = [1.0] * max(len(symbols), 1)
    
    else:
        st.sidebar.markdown("**Weights (Long-Only)**")
        weights = []
        for i, s in enumerate(symbols):
            weights.append(
                st.sidebar.number_input(
                f"{s}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.33,
                    step=0.01,
                    key=f"w_{i}_{s}"   # ✅ unique même si ticker répété
                )
            )   
        

        
        # Compteur : somme des poids
        total_w = float(sum(weights)) if len(weights) else 0.0
        st.sidebar.markdown("---")
        st.sidebar.write(f"**Sum Of Weights : {total_w:.2f}**")

        # Short message if total_w != 1
        eps = 1e-6
        if total_w < 1.0 - eps:
            st.sidebar.caption("⚠️ Sum asset weights < 1: implicit cash → distorted simulation.")
        elif total_w > 1.0 + eps:
            st.sidebar.caption("⚠️ Sum asset weights > 1: implicit leverage → distorted simulation.")


    # --- Navigation
    choice = st.selectbox(
        "Choose Analysis Page",
        [
            "Overview",
            "Data & Downloads",
            "Returns & Volatility",
            "Correlations",
            "Portfolio Optimization (Markowitz)",
        ],
    )

    # --- Load data once (used by all pages)
    prices = None
    w = None
    if len(symbols) >= 1:
        try:
            prices = get_price_series(symbols, str(start), str(end), interval=interval)
            prices = prices.dropna(how="all").ffill().dropna(how="all")
        except Exception as e:
            st.error(f"Data Retrieval Error (yfinance): {e}")

    if prices is not None and prices.shape[1] >= 1:
        # Keep only columns with enough data
        valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] > 20]
        prices = prices[valid_cols]
        symbols = valid_cols

    if prices is not None and len(symbols) >= 1:
        try:
            if weight_mode == "Equal Weight":
                w = pd.Series(1.0 / len(symbols), index=symbols)
            else:
                # align custom weights to actual columns
                w = normalize_weights(symbols, weights[:len(symbols)])
        except Exception as e:
            st.error(f"Error in Weights: {e}")
            w = None

    # -------------------------
    # Pages
    # -------------------------
    if choice == "Overview":
        st.write("Presentation Of The Portfolio Module: Multi-Assets, Simulation, Correlations, Comparison Assets vs Portfolio.")

        if prices is None or w is None or len(symbols) < 3:
            st.info("Enter At Least 3 Valid Tickers To Display The Analysis.")
            return

        # Portfolio value with rebalancing
        port_value = rebalance_portfolio(prices, w, rebalance=reb, initial_value=100.0)
        stats = portfolio_summary(prices, w, rf=rf)
        div = diversification_indicators(stats["rets"], w)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annual Return", f"{stats['ann_return']:.2%}")
        c2.metric("Annual Volatility", f"{stats['ann_vol']:.2%}")
        c3.metric("Sharpe", f"{stats['sharpe']:.2f}")
        c4.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")

        c5, c6 = st.columns(2)
        c5.metric("Average Correlation (Pairwise)", "—" if np.isnan(div["avg_pairwise_corr"]) else f"{div['avg_pairwise_corr']:.2f}")
        c6.metric("Effective Number of Assets", f"{div['effective_n']:.2f}")

        st.plotly_chart(plot_main(prices, port_value), use_container_width=True)

        with st.expander("Portfolio Weights"):
            st.dataframe(w.to_frame("Weight (%)"))

    elif choice == "Data & Downloads":
        st.subheader("Raw Data:")
        st.write("If the simulation dates are wide, not all data is displayed. You can download the full dataset as a CSV file.")
        if prices is None:
            st.info("No data To Display (Tickers Or Dates Probably Missing).")
            return

        st.write("Overview :")
        st.dataframe(prices.tail(30))

        st.download_button(
            "Download CSV",
            data=prices.to_csv().encode("utf-8"),
            file_name="prices.csv",
            mime="text/csv",
        )

    elif choice == "Returns & Volatility":
        st.subheader("Returns & Volatility")

        if prices is None or w is None or len(symbols) < 3:
            st.info("Enter At Least 3 Valid Tickers To Display The Analysis.")
            return

        rets = compute_returns(prices)
        mean = annualize_return(rets.mean())
        vol = annualize_vol(rets.std())

        st.markdown("### By Asset")
        out = pd.DataFrame({"Ann.Return": mean, "Ann.Vol": vol}).sort_values("Ann.Vol", ascending=False)
        st.dataframe(out.style.format({"Ann.Return": "{:.2%}", "Ann.Vol": "{:.2%}"}))

        st.markdown("### Portfolio")
        stats = portfolio_summary(prices, w, rf=rf)
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Return", f"{stats['ann_return']:.2%}")
        c2.metric("Annual Volatility", f"{stats['ann_vol']:.2%}")
        c3.metric("Sharpe", f"{stats['sharpe']:.2f}")

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stats["nav"].index, y=stats["nav"].values, mode="lines", name="NAV (base 100)"))
        fig.update_layout(title="Cumulated Value (Base 100)", height=420, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Correlations":
        st.subheader("Correlations")

        if prices is None or len(symbols) < 3:
            st.info("Enter At Least 3 Valid Tickers To Display The Analysis.")
            return

        rets = compute_returns(prices)
        corr = rets.corr()

        st.plotly_chart(plot_corr_heatmap(corr), use_container_width=True)

        with st.expander("Correlation Table"):
            st.dataframe(corr.style.format("{:.2f}"))

    elif choice == "Portfolio Optimization (Markowitz)":
        st.subheader("Portfolio Optimization (Markowitz)")

        if prices is None or len(symbols) < 3:
            st.info("Enter At Least 3 Valid Tickers To Display The Analysis.")
            return

        rets = compute_returns(prices)
        n_ports = st.slider("Number Of Simulated Portfolios", min_value=1000, max_value=20000, value=5000, step=1000)

        df_mc = markowitz_monte_carlo(rets, rf=rf, n_portfolios=int(n_ports), seed=42)
        best = df_mc.loc[df_mc["sharpe"].idxmax()]
        minvol = df_mc.loc[df_mc["ann_vol"].idxmin()]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Sharpe", f"{best['sharpe']:.2f}")
        c2.metric("Best Return", f"{best['ann_return']:.2%}")
        c3.metric("Best Vol", f"{best['ann_vol']:.2%}")

        st.plotly_chart(plot_frontier(df_mc), use_container_width=True)

        st.markdown("### Weight (Best Sharpe)")
        w_best = best[symbols].astype(float)
        w_best = w_best / w_best.sum()
        st.dataframe(w_best.to_frame("Weight (%)").style.format("{:.2%}"))

        st.markdown("### Weight (Min Vol)")
        w_minvol = minvol[symbols].astype(float)
        w_minvol = w_minvol / w_minvol.sum()
        st.dataframe(w_minvol.to_frame("Weight (%)").style.format("{:.2%}"))
