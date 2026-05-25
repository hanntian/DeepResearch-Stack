from retrieval.bm25 import BM25
from retrieval.fusion import DEFAULT_K_RRF, reciprocal_rank_fusion
from retrieval.qdrant_store import QdrantVectorStore


"""
Hybrid retrieval: BM25 (稀疏 / 关键词) + Vector (稠密 / 语义) + RRF 融合。

Pipeline:

    query ──┬──> BM25.top_k_retrieve(query, N)        ──> ranked list A
            │
            └──> Qdrant.top_k_retrieve(query, N)      ──> ranked list B
                                ↓
                  RRF(ranked_lists=[A, B], k_rrf=60)
                                ↓
                       取融合后 top-k 返回

设计要点：
1. 召回阶段每一路取 N (candidate_n)，远大于最终 k。这样只在一路里
   出现的 chunk 也有机会被打捞。N 太小召回崩，太大融合慢；50~100 是甜点。
2. 两路必须吃同一份 chunks 才能 chunk_id 对齐。upsert_chunks 负责这件事。
3. 融合用 RRF 而不是 weighted sum，理由见 fusion.py 顶部注释。
"""


DEFAULT_CANDIDATE_N = 50


class HybridRetriever:
    """
    组合 BM25 + QdrantVectorStore，对外暴露和单路 retriever 一样的 API：
        - upsert_chunks(chunks)
        - top_k_retrieve(query, k) -> list[dict]
    返回 dict schema 与 BM25 / QdrantVectorStore 一致，多一个 '_fusion' 字段。
    """

    def __init__(
        self,
        bm25: BM25 | None = None,
        vector_store: QdrantVectorStore | None = None,
        k_rrf: int = DEFAULT_K_RRF,
        candidate_n: int = DEFAULT_CANDIDATE_N,
    ):
        # 允许外部注入实例，默认自己 new 一份。注入主要是为了测试时塞 mock。
        self.bm25 = bm25 if bm25 is not None else BM25()
        self.vector_store = vector_store if vector_store is not None else QdrantVectorStore()
        self.k_rrf = k_rrf
        self.candidate_n = candidate_n

    # ---------- 索引构建 ----------

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        把同一份 chunks 喂给两路。
        chunk_id 必须在每条 chunk 里事先填好，否则融合时对不上。
        qa_pipeline.build_chunks 已经做了这件事。
        """
        self.bm25.upsert_chunks(chunks)
        self.vector_store.upsert_chunks(chunks)

    # ---------- 检索 ----------

    def top_k_retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_n: int | None = None,
    ) -> list[dict]:
        n = candidate_n if candidate_n is not None else self.candidate_n

        bm25_hits = self.bm25.top_k_retrieve(query, k=n)
        vector_hits = self.vector_store.top_k_retrieve(query, k=n)

        fused = reciprocal_rank_fusion(
            [bm25_hits, vector_hits],
            k_rrf=self.k_rrf,
        )
        return fused[:k]


# ===== Example usage =====
# from retrieval.chunking import chunk_documents
# import json
# from pathlib import Path
#
# raw_documents_path = Path(__file__).resolve().parents[1] / "data" / "raw_documents.jsonl"
# with raw_documents_path.open(encoding="utf-8") as f:
#     documents = [json.loads(line) for line in f if line.strip()]
# chunks = chunk_documents(documents)
# # 注意：chunk 里要有 chunk_id 才能正确融合，qa_pipeline.build_chunks 已经填好
#
# hybrid = HybridRetriever()
# hybrid.upsert_chunks(chunks)
#
# results = hybrid.top_k_retrieve("Why is vLLM faster than standard inference?", k=5)
# for r in results:
#     print(round(r["score"], 4), "|", r["title"])
#     print("  fusion:", r["_fusion"])
