import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim, index_path="data/cache/faiss.index", meta_path="data/cache/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path

        os.makedirs("data/cache", exist_ok=True)

        # -----------------------------
        # LOAD EXISTING INDEX
        # -----------------------------
        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("⚡ Loading FAISS index from disk...")
            self.index = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                self.texts = pickle.load(f)

        else:
            print("⏳ Creating new FAISS index...")
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.texts = []

    # -----------------------------
    # ADD + SAVE
    # -----------------------------
    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")

        # 🔥 Normalize (VERY IMPORTANT)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.texts.extend(texts)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)

        print("FAISS index saved to disk")

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        return results