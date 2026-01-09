# Quant Dashboard – Python Quant Project (A4 – S1)

## 1. Project Overview

This repository contains an interactive **quantitative finance dashboard** developed as part of the *Python Quant Project (A4 – Semester 1)*.

The application is built using **Python** and **Streamlit**, and provides a modular dashboard composed of two independent yet integrated components:

- **Quant A – Single Asset Analysis (Univariate)**
- **Quant B – Portfolio Management (Multivariate)**

Each module was developed independently by a different student, following a strict GitHub collaboration workflow, and later integrated into a single, functional dashboard.

---

## 2. Project Objectives

The objectives of this project are to:

- Retrieve financial market data from a **dynamic external source**
- Automatically refresh data at a **fixed frequency**
- Implement **quantitative trading strategies**
- Compute and visualize **performance and risk metrics**
- Provide a **professional, interactive dashboard**
- Apply **proper software engineering and GitHub collaboration practices**

---

## 3. Repository Structure

```bash
Python-Quant-Project-S1-A4/
│
├── app/
│   ├── main_app.py              # Streamlit entry point
│   │
│   ├── single_asset/            # Quant A module
│   │   └── ui.py
│   │
│   ├── portfolio/               # Quant B module
│   │   └── ui.py
│
├── requirements.txt             # Project dependencies
├── README.md
└── .venv/                       # Local virtual environment (not tracked)

```

The file `main_app.py` is the global entry point and allows switching between Quant A and Quant B through the Streamlit sidebar.

---

## 4. Quant A – Single Asset Analysis

### 4.1 Scope

The Quant A module focuses on the analysis of **one financial asset at a time**, such as:

- Equities (e.g. META, AAPL, ENGI.PA)
- Foreign exchange rates (e.g. EURUSD=X)
- Cryptocurrencies (e.g. BTC-USD)

Market data is retrieved dynamically from **Yahoo Finance** using the `yfinance` library.

---

### 4.2 Implemented Strategies

The following trading strategies are implemented and backtested:

- Buy & Hold
- Simple Moving Average (SMA) Crossover
- RSI Mean Reversion
- Bollinger Bands Mean Reversion
- Donchian Channel Breakout

All strategies are implemented in a **long-only** framework, without transaction costs.

---

### 4.3 User Interaction

The user can interactively configure:

- Asset ticker
- Date range
- Time interval (e.g. 1D or 1H)
- Strategy selection
- Strategy-specific parameters (windows, thresholds, etc.)

All inputs are validated to prevent invalid configurations.

---

### 4.4 Visualizations

The Quant A module provides:

- Display of the **current asset price**
- A main chart combining:
  - Raw or normalized asset price
  - Strategy equity curves
- Dedicated diagnostic charts per strategy:
  - Technical indicators (SMA, RSI, Bollinger Bands, Donchian Channels)
  - Trading positions
  - Strategy equity curves

Charts are rendered using **Plotly** to ensure clarity and robustness.

---

### 4.5 Performance Metrics

For each strategy, the following metrics are computed:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown

Metrics are displayed in a structured comparison table.

---

## 5. Quant B – Portfolio Management

The Quant B module focuses on **portfolio-level analysis**, including:

- Multi-asset portfolio construction
- Portfolio performance evaluation
- Risk and return analysis

This module was developed independently in a separate GitHub branch and later integrated via a pull request.

---

## 6. Data Source and Refresh Logic

- Market data is retrieved from **Yahoo Finance**
- The dashboard automatically refreshes every **5 minutes**
- API calls are cached for **5 minutes** to limit unnecessary requests
- Intraday data availability depends on Yahoo Finance

---

## 7. Robustness and Error Handling

The application is designed to be robust:

- Invalid tickers are handled gracefully
- Empty or insufficient datasets trigger informative warnings
- Strategy parameters are validated before execution
- Failed API calls do not crash the application

---

## 8. GitHub Collaboration Workflow

The project strictly follows the required collaboration rules:

- Separate branches for each module:
  - `feature/single-asset` (Quant A)
  - `feature/portfolio` (Quant B)
- Independent development of each module
- Integration through **Pull Requests**
- Explicit conflict resolution when required
- Clear separation of responsibilities visible in commit history

The final `main` branch contains the fully integrated and validated dashboard.

---

## 9. Installation Instructions

### 9.1 Clone the repository

```bash
git clone https://github.com/PL6-P1/Python-Quant-Project-S1-A4.git
cd Python-Quant-Project-S1-A4
```

9.2 Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

9.3 Install dependencies

```bash
pip install -r requirements.txt
```

## 10. Running the Application

From the project root directory, run:

```bash
streamlit run app/main_app.py
```

If module resolution issues occur, use:
```bash
python -m streamlit run app/main_app.py
```
The dashboard will open automatically in your default web browser.

## 11. Deployment on Linux (AWS EC2)

The application is deployed on an Ubuntu Linux virtual machine and runs 24/7 using a systemd service.

A template is provided in:
```bash
deploy/quant-dashboard.service
```

To enable the service on the VM:

```bash
sudo cp deploy/quant-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-dashboard
sudo systemctl status quant-dashboard
```

The dashboard is accessible via:
```bash
http://<EC2_PUBLIC_IP>:8501
```

## 12. Daily Report Generation (Cron)

A daily report is generated automatically using cron.

### 12.1 Cron configuration

An example cron entry is provided in:
```bash
deploy/crontab.example
```

Example (daily at 20:00):
```bash
0 20 * * * cd /home/ubuntu/Python-Quant-Project-S1-A4 && /home/ubuntu/Python-Quant-Project-S1-A4/venv/bin/python scripts/daily_report.py --ticker META --interval 1d >> logs/daily_report.log 2>&1
```

### 12.2 Outputs

Reports are stored in:
```bash
reports/
```

Logs are stored in:
```bash
logs/daily_report.log
```

## 13. Dependencies

The project relies on the following Python libraries:

```bash
streamlit
yfinance
pandas
numpy
plotly
streamlit-autorefresh
```

All dependencies are listed in requirements.txt.

## 14. Authors and Division of Work

Quant A – Single Asset Analysis: Kevin Pathmasri

Quant B – Portfolio Management: Paul Levet

Each module was developed independently in separate GitHub branches and integrated through pull requests, in strict accordance with the collaboration requirements.




Thank you! If you have any questions: 

kevin.pathmasri@edu.devinci.fr
paul.levet@edu.devinci.fr
