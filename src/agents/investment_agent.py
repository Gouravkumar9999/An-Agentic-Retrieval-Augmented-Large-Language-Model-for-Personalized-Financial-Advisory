from src.agents.llm import generate_response

def investment_agent(user_data, market_data, context, metrics):
    """
    Generates investment advice based on risk profile, market data, and
    retrieved financial knowledge.
    """
    prompt = f"""
You are a professional investment advisor.

STRICT RULES:
- USE final_risk EXACTLY: {user_data['final_risk']}
- DO NOT override risk level
- DO NOT invent numbers
- Base your advice on CONTEXT if relevant (market trends, inflation, interest rates)

METRICS:
{metrics}

FINAL RISK: {user_data['final_risk']}

LOGIC:
- LOW → 60% bonds, 30% index, 10% cash
- MEDIUM → 40% index, 30% stocks, 20% bonds, 10% cash
- HIGH → 70% stocks

CONTEXT:
{context}

TASK:
1. Recommend an allocation based on FINAL RISK and context.
2. Give short advice in simple words.

OUTPUT:
- Bullet points only
- Max 80 words
- Include Risk Level, Allocation, Advice
"""
    return generate_response(prompt)