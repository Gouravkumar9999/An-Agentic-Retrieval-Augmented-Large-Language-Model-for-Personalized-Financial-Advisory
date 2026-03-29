from src.rag.embedder import embed_texts, embed_query
from src.rag.vector_store import VectorStore
from src.rag.reranker import rerank

class RAGSystem:
    def __init__(self, documents):
        self.documents = documents

        # chunk docs
        self.chunks = self.chunk_documents(documents)

        # embed
        embeddings = embed_texts(self.chunks)

        # store
        self.store = VectorStore(len(embeddings[0]))
        self.store.add(embeddings, self.chunks)

    def chunk_documents(self, docs, chunk_size=100):
        chunks = []
        for doc in docs:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                chunks.append(chunk)
        return chunks

    def retrieve(self, query, k=3):
        q_emb = embed_query(query)
        intial_results = self.store.search(q_emb, k=10)
        reranked_results = rerank(query, intial_results, top_k=k)
        return reranked_results