from src.agents.llm import generate_response

def investment_agent(user_data, market_data, context):

    prompt = f"""
    You are an investment advisor.

    User Data:
    {user_data}

    Market Data Summary:
    {market_data}

    Context:
    {context}

    Task:
    - Suggest investment strategy
    - Recommend risk level
    - Consider inflation and market trends

    Give practical advice.
    """

    return generate_response(prompt)