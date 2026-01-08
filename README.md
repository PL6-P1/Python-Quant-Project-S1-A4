# Python-Quant-Project-S1-A4
“Projet Python / Git / Linux – Dashboard Streamlit + backtests”

Overview

This project is a quantitative finance dashboard developed in Python as part of a university assignment.
It simulates a professional quantitative research workflow within an asset management context.

The application retrieves financial data, applies quantitative strategies, and displays results through an interactive dashboard.

The project was developed collaboratively using GitHub, with a clear separation of responsibilities across branches.

app/
├── common/          # Shared data fetching utilities
├── single_asset/    # Quant A – Single Asset Analysis module
├── portfolio/       # Quant B – Multi-Asset Portfolio module
├── main_app.py      # Main Streamlit application
cron/                # Daily report generation scripts
requirements.txt
README.md

Modules

Quant A – Single Asset Analysis

- Analysis of one asset at a time
- - Implementation of backtesting strategies
- Performance metrics (returns, drawdown, risk)
- Interactive strategy parameters
- Daily report generation via cron

Quant B – Portfolio Analysis

- Multi-asset portfolio simulation (≥ 3 assets)
- Equal-weight and custom-weight allocations
- Portfolio performance and risk metrics
- Diversification and correlation analysis
- Visual comparison between assets and portfolio performance

Dashboard

- Built with Streamlit
- Interactive controls for asset selection and parameters
- Time series visualizations combining raw prices and strategy results
- Automatic data refresh

Data & Automation

- Financial data retrieved from public online sources
- Daily reports generated automatically using cron
- Scripts and cron configuration included in the repository

Collaboration & Git Workflow

- main branch: stable integrated version
- feature/single-asset: Quant A development
- feature/portfolio: Quant B development
- Pull Requests used for integration
- Clean commit history with clear separation of tasks

Installation & Execution

pip install -r requirements.txt
streamlit run app/main_app.py

Authors

Quant A – Single Asset Module - Kevin Pathmasri
Quant B – Portfolio Module - Paul Levet
