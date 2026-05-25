from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from retrieval.embedder import Embedder


COLLECTION_NAME = "deepresearch_chunks"
VECTOR_SIZE = 384

"""
Qdrant 的核心结构是 collection，里面存 points，每个 point 包含 vector 和 payload metadata；
top k retrieval 就是把 query embedding 和库里的 document vectors 做相似度搜索，返回最接近的 K 个结果。
"""
class QdrantVectorStore:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.embedder = Embedder()

    def create_collection(self):
        if self.client.collection_exists(COLLECTION_NAME):
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: list[dict]):
        self.create_collection()

        points = []
        embedded_chunks = self.embedder.embed_chunks(chunks)

        for embedded_chunk in embedded_chunks:
            chunk_id = embedded_chunk.get("chunk_id", "")
            point_id = chunk_id

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedded_chunk["embedding"],
                    payload={
                        "source_id": embedded_chunk.get("source_id", ""),
                        "chunk_id": chunk_id,
                        "chunk_index": embedded_chunk.get("chunk_index", -1),
                        "title": embedded_chunk.get("title", ""),
                        "content": embedded_chunk.get("content", ""),
                        "source": embedded_chunk.get("source", ""),
                        "url": embedded_chunk.get("url", ""),
                        "date": embedded_chunk.get("date", ""),
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def top_k_retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_vector = self.embedder.embed_query(query)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        retrieved_chunks = []

        for point in results.points:
            retrieved_chunks.append(
                {
                    "score": point.score,
                    "source_id": point.payload.get("source_id", ""),
                    "chunk_id": point.payload.get("chunk_id", ""),
                    "chunk_index": point.payload.get("chunk_index", -1),
                    "title": point.payload.get("title", ""),
                    "content": point.payload.get("content", ""),
                    "source": point.payload.get("source", ""),
                    "url": point.payload.get("url", ""),
                    "date": point.payload.get("date", ""),
                }
            )

        return retrieved_chunks

# ===== Example usage =====
# from retrieval.chunking import chunks
# store = QdrantVectorStore()
# store.upsert_chunks(chunks)

# results = store.top_k_retrieve(
#     "Why is vLLM faster than standard inference?",
#     k=2,
# )

# for item in results:
#     print("score:", item["score"])
#     print("title:", item["title"])
#     print("content:", item["content"])
#     print("url:", item["url"])
#     print()



