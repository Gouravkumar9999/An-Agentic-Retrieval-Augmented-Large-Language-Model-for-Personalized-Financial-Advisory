import pandas as pd

def preprocess_transactions(df):
    df = df.copy()

    # Rename columns (PaySim specific)
    df = df.rename(columns={
        "type": "transaction_type",
        "amount": "amount",
        "oldbalanceOrg": "balance_before",
        "newbalanceOrig": "balance_after"
    })

    # Create simple features
    df["spending"] = df["amount"]
    df["is_expense"] = df["balance_after"]< df["balance_before"]

    return df


def preprocess_stock_data(stocks):
    processed = {}

    for name, df in stocks.items():
        df = df.copy()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["daily_return"] = df["close"].pct_change()

        processed[name] = df

    return processed


def preprocess_economic_data(inflation, interest):

    def fix_df(df):
        df = df.copy()

        # normalize column names
        df.columns = [col.lower() for col in df.columns]

        # rename columns properly
        df = df.rename(columns={
            "observation_date": "date"
        })

        # convert date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # rename value column (whatever it is → value)
        value_col = [col for col in df.columns if col != "date"][0]
        df = df.rename(columns={value_col: "value"})

        return df

    inflation = fix_df(inflation)
    interest = fix_df(interest)

    return inflation, interest