from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, docs):
        self.docs = docs

        # 🔥 Cross-encoder model (STRONG)
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, top_k=5):
        # Create query-doc pairs
        pairs = [(query, doc) for doc in self.docs]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]