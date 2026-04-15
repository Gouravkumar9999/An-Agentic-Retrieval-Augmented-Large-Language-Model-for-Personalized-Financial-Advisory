from src.agents.llm import generate_response

def budget_agent(user_data, context, metrics):
    prompt = f"""
You are a financial budgeting expert.

STRICT RULES:
- USE ONLY given METRICS and FEATURES
- DO NOT recompute values
- DO NOT hallucinate
- PRIORITY: METRICS > FEATURES > CONTEXT
- DO NOT explain reasoning

METRICS:
income = {metrics['income']}
expenses = {metrics['expenses']}
savings = {metrics['savings']}
expense_ratio = {metrics['expense_ratio']}
savings_ratio = {metrics['savings_ratio']}

FEATURES:
discretionary_ratio = {user_data.get("discretionary_ratio", 0)}
essential_ratio = {user_data.get("essential_ratio", 0)}
top3_ratio = {user_data.get("top3_ratio", 0)}

CONTEXT:
{context}

LOGIC:
- discretionary_ratio > 0.5 → overspending
- expense_ratio > 0.7 → high expense risk
- savings_ratio < 0.2 → poor savings

TASK:
1. Identify spending behavior
2. Identify ONE key issue
3. Suggest 2–3 actions

OUTPUT FORMAT (STRICT):
- Spending: <short insight>
- Issue: <main problem>
- Action 1: <step>
- Action 2: <step>
- Action 3: <optional step>

DO NOT add anything else.
"""
    return generate_response(prompt)