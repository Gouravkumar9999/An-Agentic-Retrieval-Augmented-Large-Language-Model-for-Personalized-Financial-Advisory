from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.agents.llm import generate_response
import json

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# EMBEDDING-BASED SCORES
# -----------------------------
def faithfulness_score(response, context):
    response_sentences = response.split(".")
    context_chunks = context.split("\n")

    scores = []

    for sent in response_sentences:
        if len(sent.strip()) < 5:
            continue

        sent_emb = model.encode(sent, convert_to_tensor=True)
        ctx_emb = model.encode(context_chunks, convert_to_tensor=True)

        sim = util.cos_sim(sent_emb, ctx_emb).max()
        scores.append(sim.item())

    return float(np.mean(scores)) if scores else 0


def relevance_score(response, query):
    response_emb = model.encode(response, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)

    return float(util.cos_sim(response_emb, query_emb).item())


def consistency_score(response):
    sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 0]

    if len(sentences) < 2:
        return 1.0

    emb = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb, emb)

    return float(sim_matrix.mean().item())


# -----------------------------
# LLM-BASED EVALUATION (
# -----------------------------
def llm_evaluation(response, context, query):
    prompt = f"""
    You are a strict evaluator for a financial AI system.

    Evaluate the response based ONLY on the given context.

    Query:
    {query}

    Retrieved Context:
    {context}

    Response:
    {response}

    Evaluate on:

    1. Faithfulness (Is response grounded in context?)
    2. Relevance (Does it answer the query?)
    3. Consistency (No contradictions?)

    STRICT RULES:
    - Penalize hallucinations heavily
    - Penalize wrong calculations
    - Be strict, not generous

    Output ONLY JSON:
    {{
        "faithfulness": score (0 to 1),
        "relevance": score (0 to 1),
        "consistency": score (0 to 1),
        "explanation": "short reason"
    }}
    """

    try:
        result = generate_response(prompt)

        # Extract JSON safely
        start = result.find("{")
        end = result.rfind("}") + 1
        json_str = result[start:end]

        parsed = json.loads(json_str)

        return parsed

    except Exception as e:
        return {
            "faithfulness": 0.5,
            "relevance": 0.5,
            "consistency": 0.5,
            "explanation": "LLM evaluation failed"
        }


# -----------------------------
# FINAL COMBINED EVALUATION
# -----------------------------
def evaluate_response(response, context, query):
    # --- Embedding scores ---
    emb_scores = {
        "faithfulness": round(faithfulness_score(response, context), 3),
        "relevance": round(relevance_score(response, query), 3),
        "consistency": round(consistency_score(response), 3)
    }

    # --- LLM scores ---
    llm_scores = llm_evaluation(response, context, query)

    # --- Combine (weighted) ---
    final_scores = {
        "faithfulness": round((emb_scores["faithfulness"] + llm_scores["faithfulness"]) / 2, 3),
        "relevance": round((emb_scores["relevance"] + llm_scores["relevance"]) / 2, 3),
        "consistency": round((emb_scores["consistency"] + llm_scores["consistency"]) / 2, 3),
    }

    return {
        "final_scores": final_scores,
        "embedding_scores": emb_scores,
        "llm_scores": llm_scores
    }