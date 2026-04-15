from src.agents.llm import generate_response

def risk_agent(user_data, context, metrics, computed_risks):
    prompt = f"""
You are a financial risk analyst.

STRICT RULES:
- USE ONLY given values
- DO NOT hallucinate
- VALIDATE all numeric comparisons BEFORE output
- DO NOT make incorrect comparisons (e.g., 28 < 6 is FALSE)
- DO NOT explain reasoning
- PRIORITIZE COMPUTED RISKS if provided

METRICS:
expense_ratio = {metrics['expense_ratio']}
emergency_months = {metrics['emergency_months']}

FEATURES:
discretionary_ratio = {user_data.get("discretionary_ratio", 0)}
volatility = {user_data.get("volatility", 0)}

COMPUTED RISKS (USE THESE AS SOURCE OF TRUTH):
emergency_risk = {computed_risks.get("emergency_risk")}
expense_risk = {computed_risks.get("expense_risk")}
behavioral_risk = {computed_risks.get("behavioral_risk")}
volatility_risk = {computed_risks.get("volatility_risk")}

FINAL RISK: {user_data['final_risk']}

RISK RULES (STRICT IF-ELSE):

IF emergency_months < 3 → HIGH emergency risk
ELSE IF emergency_months < 6 → MEDIUM emergency risk
ELSE → LOW emergency risk

IF expense_ratio > 0.7 → HIGH expense risk
IF discretionary_ratio > 0.6 → behavioral risk
IF volatility > 300 → unstable condition

SELF-CHECK (MANDATORY):
- Check ALL comparisons are correct
- Ensure risks MATCH computed_risks values
- If mismatch → CORRECT using computed_risks

TASK:
1. Identify EXACTLY 3 risks
2. MUST align with COMPUTED RISKS
3. Each must include reason using GIVEN values
4. Prefer highest severity risks first
5. End with overall risk

OUTPUT FORMAT (STRICT):
- Risk 1: <text>
- Risk 2: <text>
- Risk 3: <text>
- Overall Risk: {user_data['final_risk']}

DO NOT add anything else.
"""
    return generate_response(prompt)