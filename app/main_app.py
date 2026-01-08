# app/main_app.py


import streamlit as st
from app.single_asset.ui import run_single_asset
from app.portfolio.ui import run_portfolio

st.set_page_config(page_title="Quant Dashboard", layout="wide")

module = st.sidebar.radio("Module", ["Single Asset Management", "Portfolio Management"])

if module == "Single Asset Management":
    run_single_asset()
else:
    run_portfolio()
