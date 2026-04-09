from src.agents.llm import generate_response

def budget_agent(user_data, context, metrics):
    """
    Generates budgeting advice based on user metrics, spending behavior,
    and retrieved financial knowledge.
    """
    prompt = f"""
You are a financial budgeting expert.

STRICT RULES:
- NEVER recompute values.
- USE ONLY the provided metrics and extra features.
- DO NOT hallucinate or invent numbers.
- Always refer to the provided CONTEXT if relevant.

METRICS:
{metrics}

EXTRA FEATURES:
discretionary_ratio = {user_data.get("discretionary_ratio", 0)}
top3_ratio = {user_data.get("top3_ratio", 0)}
essential_ratio = {user_data.get("essential_ratio", 0)}

CONTEXT:
{context}

TASK:
1. Evaluate spending behavior.
2. Identify overspending or optimization opportunities.
3. Suggest concise actionable steps.

OUTPUT:
- Bullet points only
- Max 80 words
- Highlight overspending if discretionary_ratio > 0.5
- Otherwise suggest optimization
"""
    return generate_response(prompt)