import pandas as pd
import os

DATA_PATH = "data/raw"

def load_transactions(path="data/raw/transactions.csv"):
    df = pd.read_csv(path)

    # Debug (remove later)
    print("Before rename:", df.columns)

    # Safe rename
    if "category" in df.columns:
        df.rename(columns={"category": "transaction_type"}, inplace=True)

    print("After rename:", df.columns)

    # Type conversions
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_stock_data():
    stocks = {}

    for file in ["aapl.csv", "msft.csv", "spy.csv"]:
        path = os.path.join(DATA_PATH, file)

        df = pd.read_csv(path, skiprows=3, header=None)
        df.columns = ["date", "close", "high", "low", "open", "volume"]

        stocks[file] = df

    return stocks


def load_economic_data():
    inflation = pd.read_csv(os.path.join(DATA_PATH, "inflation.csv"))
    interest = pd.read_csv(os.path.join(DATA_PATH, "interest_rate.csv"))
    return inflation, interest


def load_knowledge_base():
    kb_path = os.path.join(DATA_PATH, "knowledge")
    documents = []

    for file in os.listdir(kb_path):
        with open(os.path.join(kb_path, file), "r") as f:
            documents.append(f.read())

    return documents