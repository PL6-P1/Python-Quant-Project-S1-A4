import streamlit as st
from single_asset.ui import run_single_asset
from portfolio.ui import run_portfolio

st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.title("Quant Dashboard - Projet Python / Git / Linux")

mode = st.sidebar.selectbox(
    "Choisir le module",
    ["Single asset (Quant A)", "Portefeuille (Quant B)"]
)

if mode.startswith("Single"):
    run_single_asset()
else:
    run_portfolio()