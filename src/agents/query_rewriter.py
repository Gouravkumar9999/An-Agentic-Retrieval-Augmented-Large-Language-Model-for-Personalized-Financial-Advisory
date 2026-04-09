from src.agents.llm import generate_response

def rewrite_query(query):
    prompt = f"""
    You are a query optimizer for a financial RAG system.

    Convert the user query into a more detailed and retrieval-friendly query.

    RULES:
    - Keep meaning same
    - Add relevant financial keywords
    - Make it more specific
    - Do NOT answer the query

    Original Query:
    {query}

    Rewritten Query:
    """

    try:
        rewritten = generate_response(prompt)
        return rewritten.strip()
    except:
        return query  