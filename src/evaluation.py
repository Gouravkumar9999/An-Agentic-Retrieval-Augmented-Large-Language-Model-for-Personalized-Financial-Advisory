from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def faithfulness_score(response, context):

    response_emb = model.encode(response, convert_to_tensor=True)
    context_emb = model.encode(context, convert_to_tensor=True)

    similarity = util.cos_sim(response_emb, context_emb)

    return float(similarity.mean())
def relevance_score(response, query):

    response_emb = model.encode(response, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)

    similarity = util.cos_sim(response_emb, query_emb)

    return float(similarity.mean())
def consistency_score(response):

    sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 0]

    if len(sentences) < 2:
        return 1.0

    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    return float(sim_matrix.mean())
def evaluate_response(response, context, query):

    return {
        "faithfulness": faithfulness_score(response, context),
        "relevance": relevance_score(response, query),
        "consistency": consistency_score(response)
    }