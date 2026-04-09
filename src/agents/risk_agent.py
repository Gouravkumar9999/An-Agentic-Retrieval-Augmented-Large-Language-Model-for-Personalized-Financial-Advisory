from src.agents.llm import generate_response

def risk_agent(user_data, context, metrics):
    """
    Analyzes financial risks based on spending behavior, volatility, and
    emergency funds.
    """
    prompt = f"""
You are a financial risk analyst.

STRICT RULES:
- USE ONLY the provided metrics and features.
- DO NOT contradict any values.
- Base your reasoning on CONTEXT if relevant.

METRICS:
{metrics}

FEATURES:
discretionary_ratio = {user_data.get("discretionary_ratio", 0)}
volatility = {user_data.get("volatility", 0)}

FINAL RISK: {user_data['final_risk']}

LOGIC:
- expense_ratio < 0.3 → LOW risk
- emergency_months > 6 → LOW risk
- discretionary_ratio > 0.6 → behavioral risk
- high volatility → unstable

TASK:
1. Identify 3 main risks.
2. Provide a brief reason for each.
3. End with Overall Risk: {user_data['final_risk']}

OUTPUT:
- Bullet points only
- Max 80 words
"""
    return generate_response(prompt)