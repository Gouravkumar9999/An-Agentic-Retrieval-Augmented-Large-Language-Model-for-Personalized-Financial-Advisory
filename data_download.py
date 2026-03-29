import yfinance as yf
import pandas as pd

# Download stock data
stocks = ["AAPL", "MSFT", "SPY"]

for stock in stocks:
    data = yf.download(stock, start="2015-01-01", end="2024-01-01")
    data.to_csv(f"{stock}.csv")
    print(f"{stock} downloaded")