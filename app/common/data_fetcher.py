# app/common/data_fetcher.py
from __future__ import annotations

from typing import List, Union
import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False


def _download_yf(symbols: List[str], start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    # yfinance returns different shapes depending on len(symbols)
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Adj Close; fallback Close
        if ("Adj Close" in df.columns.get_level_values(0)):
            out = df["Adj Close"].copy()
        elif ("Close" in df.columns.get_level_values(0)):
            out = df["Close"].copy()
        else:
            raise ValueError("Impossible de trouver 'Adj Close' ou 'Close' dans les données téléchargées.")
    else:
        # single ticker case -> Series-like columns
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            raise ValueError("Impossible de trouver 'Adj Close' ou 'Close' dans les données téléchargées.")
        out = df[[col]].rename(columns={col: symbols[0]})

    out = out.sort_index()
    out = out.dropna(how="all")
    # Ensure all columns exist
    for s in symbols:
        if s not in out.columns:
            out[s] = pd.NA

    return out


if _HAS_STREAMLIT:
    @st.cache_data(ttl=300, show_spinner=False)  # refresh cache every 5 minutes
    def get_price_series(symbols: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(symbols) < 1:
            raise ValueError("Veuillez fournir au moins un ticker.")
        return _download_yf(symbols, start, end, interval)
else:
    def get_price_series(symbols: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        if len(symbols) < 1:
            raise ValueError("Veuillez fournir au moins un ticker.")
        return _download_yf(symbols, start, end, interval)
