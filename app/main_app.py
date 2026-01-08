# app/main_app.py
"""
Main entrypoint for the Quant Dashboard.

This file is responsible for:
- configuring Streamlit
- routing between Quant A (Single Asset) and Quant B (Portfolio)
- ensuring the project root is correctly added to PYTHONPATH so that
  imports like `from app.single_asset.ui import ...` work reliably
  when running `streamlit run app/main_app.py`
"""

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# This avoids: ModuleNotFoundError: No module named 'app'
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Standard imports
# ------------------------------------------------------------
import streamlit as st

from app.single_asset.ui import run_single_asset
from app.portfolio.ui import run_portfolio

# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Quant Dashboard",
    layout="wide",
)

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Select module",
    [
        "Single Asset Management",   # Quant A
        "Portfolio Management",      # Quant B
    ],
)

# ------------------------------------------------------------
# Main routing
# ------------------------------------------------------------
if module == "Single Asset Management":
    run_single_asset()
else:
    run_portfolio()
