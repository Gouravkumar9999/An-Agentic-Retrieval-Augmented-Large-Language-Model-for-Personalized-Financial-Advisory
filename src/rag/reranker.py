from sentence_transformers import SentenceTransformer, util

rerank_model = SentenceTransformer("all-MiniLM-L6-v2")

def rerank(query, docs, top_k=5):

    query_emb = rerank_model.encode(query, convert_to_tensor=True)
    doc_embs = rerank_model.encode(docs, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]