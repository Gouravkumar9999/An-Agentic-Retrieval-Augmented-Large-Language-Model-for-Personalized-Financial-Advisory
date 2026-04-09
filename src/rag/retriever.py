import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.embedder import embed_texts, embed_query
from src.rag.vector_store import VectorStore
from src.rag.reranker import Reranker


class RAGSystem:
    def __init__(self, documents):
        self.documents = documents

        os.makedirs("data/cache", exist_ok=True)

        emb_path = "data/cache/embeddings.npy"
        chunks_path = "data/cache/chunks.pkl"

        # -----------------------------
        # LOAD / CREATE CHUNKS
        # -----------------------------
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.chunks = self.chunk_documents(documents)
            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)

        # -----------------------------
        # BM25 SETUP
        # -----------------------------
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        # -----------------------------
        # LOAD / CREATE EMBEDDINGS
        # -----------------------------
        if os.path.exists(emb_path):
            print("⚡ Loading cached embeddings...")
            embeddings = np.load(emb_path)
        else:
            print("⏳ Creating embeddings...")
            embeddings = embed_texts(self.chunks)
            np.save(emb_path, embeddings)

        # -----------------------------
        # VECTOR STORE (FAISS)
        # -----------------------------
        self.store = VectorStore(len(embeddings[0]))

        if self.store.index.ntotal == 0:
            print("📦 Populating FAISS index...")
            self.store.add(embeddings, self.chunks)
        else:
            print("✅ Using existing FAISS index")

    # -----------------------------
    # CHUNKING
    # -----------------------------
    def chunk_documents(self, docs, chunk_size=60):
        chunks = []
        for doc in docs:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    # -----------------------------
    # HYBRID RETRIEVAL + RERANK
    # -----------------------------
    def retrieve(self, query, k=3):
        # -------- FAISS SEARCH --------
        q_emb = embed_query(query)
        vector_results = self.store.search(q_emb, k=10)

        # -------- BM25 SEARCH --------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_bm25_idx = np.argsort(bm25_scores)[::-1][:10]
        bm25_results = [self.chunks[i] for i in top_bm25_idx]

        # -------- MERGE --------
        combined = list(set(vector_results + bm25_results))

        # -------- CROSS-ENCODER RERANK --------
        reranker = Reranker(combined)
        return reranker.rerank(query, top_k=k)