from src.agents.llm import generate_response

def budget_agent(user_data, context):

    prompt = f"""
    You are a budgeting expert.

    User Data:
    {user_data}

    Context:
    {context}

    Task:
    - Analyze spending behavior
    - Suggest budgeting improvements
    - Identify unnecessary expenses

    Give structured advice.
    """

    return generate_response(prompt)