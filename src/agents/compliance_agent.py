from src.agents.llm import generate_response

def compliance_agent(report):

    prompt = f"""
    You are a financial compliance auditor.

    Evaluate the following report based on:
    1. Financial safety
    2. Risk disclosure
    3. Absence of harmful/illegal advice
    4. Real-world applicability

    Report:
    {report}

    Output STRICTLY in JSON:
    {{
        "status": "SAFE or UNSAFE",
        "issues": ["list of problems"],
        "confidence": "low/medium/high"
    }}
    """

    return generate_response(prompt)