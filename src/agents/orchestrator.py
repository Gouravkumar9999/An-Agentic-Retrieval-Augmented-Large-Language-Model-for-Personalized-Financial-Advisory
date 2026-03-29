from src.agents.budget_agent import budget_agent
from src.agents.investment_agent import investment_agent
from src.agents.risk_agent import risk_agent
from src.agents.compliance_agent import compliance_agent
history = []
def run_agents(user_data, market_data, context):
    global history
    budget = budget_agent(user_data, context)
    investment = investment_agent(user_data, market_data, context)
    risk = risk_agent(user_data, context)

    final_output = f"""
    ===== PERSONAL FINANCE REPORT =====

    📊 Budget Analysis:
    {budget}

    📈 Investment Advice:
    {investment}

    ⚠️ Risk Assessment:
    {risk}
    """
    compliance = compliance_agent(final_output)

    final_output += f"\n\n🛡 Compliance Check:\n{compliance}"

    history.append({
        "user_data": user_data,
        "context": context,
        "report": final_output
    })
    return final_output