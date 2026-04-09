def analyze_transactions(df):
    df = df.copy()

    df["category"] = df["transaction_type"]

    # Monthly spend (robust)
    monthly_series = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()

    monthly_spend = monthly_series.mean() if len(monthly_series) > 0 else 0

    categories = df.groupby("category")["amount"].sum().to_dict()

    return {
        "monthly_spending": round(float(monthly_spend), 2),
        "category_breakdown": {k: round(float(v), 2) for k, v in categories.items()}
    }
def generate_market_summary(inflation_df, interest_df):

    latest_inflation = inflation_df.iloc[-1]["value"]
    prev_inflation = inflation_df.iloc[-2]["value"]

    latest_rate = interest_df.iloc[-1]["value"]
    prev_rate = interest_df.iloc[-2]["value"]

    inflation_trend = "rising" if latest_inflation > prev_inflation else "falling"
    rate_trend = "increasing" if latest_rate > prev_rate else "decreasing"

    return f"Inflation is {inflation_trend}. Interest rates are {rate_trend}."