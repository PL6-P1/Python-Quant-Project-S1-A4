# app/single_asset/ui.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Auto-refresh toutes les 5 minutes
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    # Si le package n'est pas installé, on définit un no-op
    def st_autorefresh(*args, **kwargs):
        return None


# ============================
#   FONCTIONS UTILITAIRES
# ============================

def compute_buy_and_hold(returns: pd.Series) -> pd.Series:
    """
    Equity curve d'une stratégie Buy & Hold.
    On suppose qu'on investit 1 au début et qu'on tient la position.
    """
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    return equity


def compute_ma_crossover(
    price: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
):
    """
    Stratégie Moving Average Crossover :
    - MA courte, MA longue
    - signal = 1 si MA courte > MA longue, 0 sinon
    - on applique ce signal aux rendements de l'actif
    """
    price = price.dropna()

    ma_short = price.rolling(short_window).mean()
    ma_long = price.rolling(long_window).mean()

    # Signal 1 si MA courte > MA longue
    signal = (ma_short > ma_long).astype(int)

    # Décalage d'un jour pour éviter le look-ahead bias
    signal = signal.shift(1).fillna(0)

    returns = price.pct_change().fillna(0)
    strat_returns = signal * returns

    equity = (1 + strat_returns).cumprod()

    return equity, ma_short, ma_long, signal, strat_returns


def compute_metrics(equity: pd.Series, returns: pd.Series) -> dict:
    """
    Calcule quelques métriques standard :
    - rendement total
    - rendement annualisé
    - volatilité annualisée
    - Sharpe ratio (rf = 0)
    - max drawdown
    """
    equity = equity.dropna()
    returns = returns.dropna()

    if len(equity) == 0 or len(returns) == 0:
        return {}

    total_return = equity.iloc[-1] - 1
    nb_days = len(returns)

    # 252 jours de bourse / an
    ann_return = (1 + total_return) ** (252 / nb_days) - 1
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "Total return": total_return,
        "Annualized return": ann_return,
        "Annualized vol": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": max_dd,
    }


# ============================
#   PAGE STREAMLIT QUANT A
# ============================

def run_single_asset():
    # Auto-refresh toutes les 5 minutes (300 000 ms)
    st_autorefresh(interval=5 * 60 * 1000, key="single_asset_refresh")

    st.header("Single Asset Analysis (Quant A)")
    st.write(
        "Analyse d'un seul actif : téléchargement des données (API Yahoo Finance), "
        "affichage des prix, backtests de stratégies simples, métriques, et mise à jour automatique."
    )

    # ----------------------------
    # 1) Paramètres utilisateur
    # ----------------------------
    st.subheader("Paramètres de l'actif")

    ticker = st.text_input("Ticker de l'actif :", "META")

    today = date.today()
    default_start = today - timedelta(days=365)

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Date de début", value=default_start, max_value=today)
    with col2:
        end = st.date_input("Date de fin", value=today, max_value=today)

    if start >= end:
        st.error("La date de début doit être strictement avant la date de fin.")
        return

    st.subheader("Paramètres de la stratégie MA Crossover")
    c1, c2 = st.columns(2)
    with c1:
        short_window = st.slider("MA courte (jours)", min_value=5, max_value=50, value=20, step=1)
    with c2:
        long_window = st.slider("MA longue (jours)", min_value=20, max_value=200, value=50, step=5)

    if short_window >= long_window:
        st.error("La MA courte doit être strictement inférieure à la MA longue.")
        return

    # ----------------------------
    # 2) Chargement des données
    # ----------------------------
    # Pas de bouton ici : la page se re-run automatiquement (auto-refresh),
    # donc on recharge systématiquement.
    if not ticker:
        st.error("Merci d'entrer un ticker valide (ex : META, AAPL, BTC-USD…).")
        return

    st.write(f"Téléchargement des données pour **{ticker}** ...")
    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        st.warning("Aucune donnée trouvée pour ce ticker / cette période.")
        return

    st.subheader("Données brutes OHLCV (dernières lignes)")
    st.dataframe(data.tail())

    # Choix de la série de prix
    if "Adj Close" in data.columns:
        price = data["Adj Close"]
    elif "Close" in data.columns:
        price = data["Close"]
    else:
        st.error("Impossible de trouver une colonne de prix ('Adj Close' ou 'Close').")
        return

    # Si jamais price est un DataFrame (MultiIndex), on prend la première colonne
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    price = price.dropna()

    if len(price) < long_window + 5:
        st.warning("Période trop courte pour calculer correctement les moyennes mobiles choisies.")
        return

    # ----------------------------
    # 3) Affichage du prix actuel
    # ----------------------------
    st.subheader("Prix actuel")

    last_price = price.iloc[-1]
    if len(price) > 1:
        prev_price = price.iloc[-2]
        delta_value = last_price - prev_price
        delta_pct = (last_price / prev_price - 1) * 100
        delta_str = f"{delta_value:.2f} ({delta_pct:+.2f}%)"
    else:
        delta_str = "N/A"

    st.metric("Last price", f"{last_price:.2f}", delta=delta_str)

    # ----------------------------
    # 4) Prix + moyennes mobiles
    # ----------------------------
    st.subheader("Prix de clôture et moyennes mobiles")

    equity_ma, ma_short, ma_long, signal, strat_returns = compute_ma_crossover(
        price, short_window=short_window, long_window=long_window
    )

    price_df = pd.DataFrame(
        {
            "Price": price,
            f"MA_{short_window}": ma_short,
            f"MA_{long_window}": ma_long,
        }
    )

    st.line_chart(price_df)

    # ----------------------------
    # 5) Backtests : Buy & Hold vs MA Crossover
    # ----------------------------
    st.subheader("Backtests : Buy & Hold vs MA Crossover")

    asset_returns = price.pct_change().dropna()
    equity_bh = compute_buy_and_hold(asset_returns)

    equity_df = pd.DataFrame(
        {
            "Buy & Hold": equity_bh,
            f"MA Crossover ({short_window}, {long_window})": equity_ma,
        }
    ).dropna()

    st.line_chart(equity_df)

    # ----------------------------
    # 6) Graphe principal combiné (prix normalisé + stratégies)
    # ----------------------------
    st.subheader("Graphe principal : prix normalisé vs stratégies")

    price_norm = price / price.iloc[0]
    combined_df = pd.DataFrame(
        {
            "Price (normalized)": price_norm,
            "Buy & Hold": equity_bh,
            f"MA Crossover ({short_window}, {long_window})": equity_ma,
        }
    ).dropna()

    st.line_chart(combined_df)

    # ----------------------------
    # 7) Métriques de performance
    # ----------------------------
    st.subheader("Métriques de performance")

    metrics_bh = compute_metrics(equity_bh, asset_returns)
    metrics_ma = compute_metrics(equity_ma, strat_returns)

    metrics_df = pd.DataFrame(
        {
            "Buy & Hold": metrics_bh,
            f"MA Crossover ({short_window}, {long_window})": metrics_ma,
        }
    ).T  # stratégies en lignes

    st.dataframe(
        metrics_df.style.format(
            {
                "Total return": "{:.2%}",
                "Annualized return": "{:.2%}",
                "Annualized vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.2%}",
            }
        )
    )
