import yfinance as yf
import pandas as pd
from datetime import datetime

def compute_drawdown(series):
    cumulative = series.cummax()
    dd = (series - cumulative) / cumulative
    return dd.min()

def generate_daily_report(ticker="META"):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"report_{ticker}_{today}.txt"

    # Télécharger les données du jour
    df = yf.download(ticker, period="5d", interval="1d")

    if df.empty:
        print("No data downloaded.")
        return

    last_row = df.iloc[-1]
    first_row = df.iloc[0]

    open_price = last_row["Open"]
    close_price = last_row["Close"]
    high = last_row["High"]
    low = last_row["Low"]
    volume = last_row["Volume"]
    daily_return = (close_price - open_price) / open_price

    # Max drawdown sur la période téléchargée
    dd = compute_drawdown(df["Adj Close"])

    report = f"""
    DAILY REPORT – {ticker}

    Date: {today}

    Open: {open_price:.2f}
    Close: {close_price:.2f}
    High: {high:.2f}
    Low: {low:.2f}
    Volume: {volume}

    Daily return: {daily_return:.2%}
    Max drawdown (last 5 days): {dd:.2%}

    Generated automatically by cron.
    """

    with open(filename, "w") as f:
        f.write(report)

    print(f"Report generated: {filename}")


if __name__ == "__main__":
    generate_daily_report("META")
