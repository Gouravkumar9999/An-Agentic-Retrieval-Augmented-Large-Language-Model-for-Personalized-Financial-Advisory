import pandas as pd
import os

DATA_PATH = "data/raw"

def load_transactions():
    path = os.path.join(DATA_PATH, "transactions.csv")
    df = pd.read_csv(path)
    return df

def load_stock_data():
    import pandas as pd
    import os

    DATA_PATH = "data/raw"
    stocks = {}

    for file in ["aapl.csv", "msft.csv", "spy.csv"]:
        path = os.path.join(DATA_PATH, file)

        # 🔥 KEY FIX
        df = pd.read_csv(path, skiprows=3, header=None)

        # 🔥 manually assign correct columns
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