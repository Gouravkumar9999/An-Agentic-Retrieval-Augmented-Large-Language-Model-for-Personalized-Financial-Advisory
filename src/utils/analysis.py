def analyze_transactions(df):
    
    # rename type → category
    df = df.copy()
    df["category"] = df["transaction_type"]

    # monthly spending (assuming step ~ time)
    monthly_spend = df["amount"].sum() / 12

    # spending per category
    categories = df.groupby("category")["amount"].sum().to_dict()

    return {
        "monthly_spending": monthly_spend,
        "category_breakdown": categories
    }
def generate_market_summary(inflation_df, interest_df):

    latest_inflation = inflation_df.iloc[-1]["value"]
    prev_inflation = inflation_df.iloc[-2]["value"]

    latest_rate = interest_df.iloc[-1]["value"]
    prev_rate = interest_df.iloc[-2]["value"]

    inflation_trend = "rising" if latest_inflation > prev_inflation else "falling"
    rate_trend = "increasing" if latest_rate > prev_rate else "decreasing"

    return f"Inflation is {inflation_trend}. Interest rates are {rate_trend}."