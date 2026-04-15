from src.agents.llm import generate_response

def investment_agent(user_data, market_data, context, metrics):
    prompt = f"""
You are a professional investment advisor.

STRICT RULES:
- USE final_risk EXACTLY: {user_data['final_risk']}
- DO NOT change allocation logic
- DO NOT hallucinate
- DO NOT explain reasoning
- OUTPUT must match format exactly

FINAL RISK: {user_data['final_risk']}

ALLOCATION RULES (STRICT):
IF LOW:
- Bonds: 60
- Index Funds: 30
- Cash: 10

IF MEDIUM:
- Index Funds: 40
- Stocks: 30
- Bonds: 20
- Cash: 10

IF HIGH:
- Stocks: 70
- Index Funds: 20
- Cash: 10

CONTEXT:
{context}

TASK:
1. Select allocation based ONLY on FINAL RISK
2. Give 1 short advice

OUTPUT FORMAT (STRICT):
- Risk Level: {user_data['final_risk']}
- Allocation:
  Index Funds: <value>%
  Stocks: <value>%
  Bonds: <value>%
  Cash: <value>%
- Advice: <one line>

DO NOT add anything else.
"""
    return generate_response(prompt)