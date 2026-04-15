import pandas as pd
import numpy as np

ESSENTIAL = ["Health", "Transport", "Fuel", "Market"]
DISCRETIONARY = ["Coffe", "Restuarant", "Travel", "Film/enjoyment", "Joy"]

def preprocess_transactions(df):
    df = df.copy()

    # Rename
    if "category" in df.columns:
        df.rename(columns={"category": "transaction_type"}, inplace=True)

    # Validate
    required_cols = ["date", "transaction_type", "amount"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} missing!")

    # Convert
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    #  REMOVE OUTLIERS
    df = df[df["amount"] < df["amount"].quantile(0.99)]

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df["month"] = df["date"].dt.to_period("M")

    total_spend = df["amount"].sum()

    # Category breakdown
    category_spend = df.groupby("transaction_type")["amount"].sum()

    #  TOP 3 concentration
    top3_ratio = category_spend.sort_values(ascending=False).head(3).sum() / total_spend

    #  ESSENTIAL / DISCRETIONARY
    essential_spend = df[df["transaction_type"].isin(ESSENTIAL)]["amount"].sum()
    discretionary_spend = df[df["transaction_type"].isin(DISCRETIONARY)]["amount"].sum()

    essential_ratio = essential_spend / total_spend if total_spend else 0
    discretionary_ratio = discretionary_spend / total_spend if total_spend else 0

    #  VOLATILITY
    monthly_spend = df.groupby("month")["amount"].sum()
    volatility = monthly_spend.std()

    # TRANSACTION FREQUENCY
    days = (df["date"].max() - df["date"].min()).days + 1
    txn_frequency = len(df) / days if days else 0

    df.attrs["features"] = {
        "top3_ratio": round(top3_ratio, 3),
        "essential_ratio": round(essential_ratio, 3),
        "discretionary_ratio": round(discretionary_ratio, 3),
        "volatility": round(volatility if not np.isnan(volatility) else 0, 3),
        "txn_frequency": round(txn_frequency, 3)
    }

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