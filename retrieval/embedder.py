from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode( 
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = dict(chunk)
            embedded_chunk["embedding"] = embedding
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()


# ========= Example Usage =========
# from retrieval.chunking import chunks

# embedder = Embedder()
# texts = [chunk["content"] for chunk in chunks]
# embeddings = embedder.embed_texts(texts)

# print(len(embeddings))
# print(len(embeddings[0])) #embedding 维度是 384
# print(embeddings[0][:5])


