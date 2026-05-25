import re
from typing import Iterable

from rank_bm25 import BM25Okapi


"""
BM25 (Okapi BM25) 关键词检索。

和 vector retrieval 的区别：
- vector：把 query 和 chunk 都映射到向量空间，比 cosine 相似度，擅长语义/同义改写。
- BM25  ：基于 token 倒排 + 词频 / 文档频率打分，擅长精确关键词、专有名词、缩写、代码符号。

在 hybrid retrieval 里这两路一般互补：
- "vLLM PagedAttention"           -> BM25 直接命中关键词；
- "为什么这种推理引擎更快"          -> vector 才能召回讲 vLLM 的 chunk。
所以同时跑两路再做分数融合 (RRF / weighted sum) 通常比单路效果好。

底层打分算法不自己写，直接用 rank_bm25 里的 BM25Okapi:
    score(D, Q) = sum_t idf(t) * f(t, D) * (k1 + 1)
                  / (f(t, D) + k1 * (1 - b + b * |D|/avgdl))
我们只负责：
1. 把 chunk -> token 列表 (tokenize)
2. 喂给 BM25Okapi 建索引
3. 用 query token 调 get_scores，包装成和 QdrantVectorStore 一致的返回结构
"""


# BM25Okapi 默认就是 k1=1.5, b=0.75，这里显式列出来方便后面调
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# 极简英文 stopword 表，避免引 nltk
# 检索里 stopword 主要是噪音，去掉让 idf 更聚焦在 informative token 上
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "at",
    "this", "that", "these", "those", "it", "its", "from", "but", "if",
    "then", "than", "so", "such", "we", "you", "they", "i", "he", "she",
    "do", "does", "did", "have", "has", "had", "not", "no",
}

# 简单 tokenizer：小写 + 按非字母数字切分
# 不做 stemming / lemmatization，保持透明；要更强可以换 nltk / spaCy
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def default_tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    if not text:
        return []
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25:
    """
    内存型 BM25 检索器，对外 API 对齐 QdrantVectorStore：
        - upsert_chunks(chunks)
        - top_k_retrieve(query, k) -> list[dict]
    返回 schema 与 QdrantVectorStore 一致，hybrid 融合时方便统一处理。
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        remove_stopwords: bool = True,
    ):
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        # 原始 chunk payload，下标和 BM25Okapi 内部 corpus 下标一一对应
        self.chunks: list[dict] = []
        # rank_bm25 的 BM25Okapi 实例；空库时为 None
        self._bm25: BM25Okapi | None = None

    # ---------- 索引构建 ----------

    def _tokenize(self, text: str) -> list[str]:
        return default_tokenize(text, remove_stopwords=self.remove_stopwords)

    def upsert_chunks(self, chunks: Iterable[dict]) -> None:
        """
        构建 BM25 索引。
        注意：rank_bm25 在构造时一次性算好 idf / avgdl，没有增量接口，
        所以这里是“整体 rebuild”，不是真正的 upsert。
        BM25 的 idf / avgdl 都是全局统计量，离线场景整体 rebuild 最稳。
        """
        chunks = list(chunks)
        self.chunks = chunks

        tokenized_corpus = [self._tokenize(c.get("content", "") or "") for c in chunks]

        # rank_bm25 要求 corpus 中每篇至少有 1 个 token，全空文档会让 avgdl=0 报错
        # 给空文档塞一个占位 token，保证不会污染其它分数（这个占位词不会在 query 里出现）
        for i, toks in enumerate(tokenized_corpus):
            if not toks:
                tokenized_corpus[i] = ["__empty__"]

        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    # ---------- 打分与检索 ----------

    def top_k_retrieve(self, query: str, k: int = 5) -> list[dict]:

        query_tokens = self._tokenize(query)

        # get_scores 返回 shape=(n_docs,) 的 numpy 数组，下标对齐 self.chunks
        scores = self._bm25.get_scores(query_tokens)

        # argsort 取 top-k；过滤掉 score<=0 的（说明 query 词一个都没命中）
        # 用 enumerate 而不是 numpy.argpartition 是为了实现简单 + corpus 体量小够用
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for i, score in ranked:
            if score <= 0:
                break
            chunk = self.chunks[i]
            results.append(
                {
                    "score": float(score),
                    "source_id": chunk.get("source_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "title": chunk.get("title", ""),
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", ""),
                    "url": chunk.get("url", ""),
                    "date": chunk.get("date", ""),
                }
            )
            if len(results) >= k:
                break
        return results

    # ---------- 状态查询的小工具 ----------

    def __len__(self) -> int:
        return len(self.chunks)


# ===== Example usage =====
# from retrieval.chunking import chunk_documents
# import json
# from pathlib import Path

# raw_documents_path = Path(__file__).resolve().parents[1] / "data" / "raw_documents.jsonl"
# with raw_documents_path.open(encoding="utf-8") as f:
#     documents = [json.loads(line) for line in f if line.strip()]

# chunks = chunk_documents(documents)

# bm25 = BM25()
# bm25.upsert_chunks(chunks)

# results = bm25.top_k_retrieve("Why is vLLM faster than standard inference?", k=3)
# for item in results:
#     print("score:", item["score"])
#     print("title:", item["title"])
#     print("content:", item["content"][:120], "...")
#     print()
