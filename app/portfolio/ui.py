
import streamlit as st

def run_portfolio():
    st.header("Module Portefeuille (Quant B)")

    choice = st.selectbox(
        "Choisir l’analyse",
        [
            "Vue d’ensemble",
            "Données & téléchargements",
            "Rendements & volatilité",
            "Corrélations",
            "Optimisation de portefeuille (Markowitz)"
        ]
    )

    if choice == "Vue d’ensemble":
        st.write("Présentation du module portefeuille.")
    elif choice == "Données & téléchargements":
        show_data()
    elif choice == "Rendements & volatilité":
        show_returns()
    elif choice == "Corrélations":
        show_correlations()
    elif choice == "Optimisation de portefeuille (Markowitz)":
        show_optimization()


import yfinance as yf
import pandas as pd

def get_price_series(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)['Adj Close']
    return data


def show_data():
    st.subheader("Données brutes")
    symbols = st.text_input("Tickers (séparés par virgules)", "AAPL, MSFT, META").split(",")
    symbols = [s.strip() for s in symbols]

    start = st.date_input("Date début")
    end = st.date_input("Date fin")

    if st.button("Télécharger"):
        from app.common.data_fetcher import get_price_series
        df = get_price_series(symbols, start, end)
        st.write(df)

def show_returns():
    st.subheader("Rendements & volatilité")

    from app.common.data_fetcher import get_price_series

    symbols = ["AAPL", "MSFT", "META"]
    df = get_price_series(symbols, "2020-01-01", "2025-01-01")

    returns = df.pct_change().dropna()
    
    st.write("Rendements moyens (%) :", (returns.mean() * 100).round(2))
    st.write("Volatilité annuelle (%) :", (returns.std() * (252**0.5) * 100).round(2))


def show_correlations():
    st.subheader("Corrélations")
    
    from app.common.data_fetcher import get_price_series

    symbols = ["AAPL", "MSFT", "META"]
    df = get_price_series(symbols, "2020-01-01", "2025-01-01")

    corr = df.pct_change().dropna().corr()
    st.dataframe(corr)
