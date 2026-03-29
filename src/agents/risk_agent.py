from src.agents.llm import generate_response

def risk_agent(user_data, context):

    prompt = f"""
    You are a financial risk analyst.

    User Data:
    {user_data}

    Context:
    {context}

    Task:
    - Identify financial risks
    - Assess stability
    - Warn about potential issues

    Be critical and analytical.
    """

    return generate_response(prompt)