from typing import Callable, Hashable, Iterable


"""
Rank fusion 工具，hybrid retrieval 用。

Reciprocal Rank Fusion (Cormack et al., 2009):

    RRF_score(d) = Σ_i  1 / (k_rrf + rank_i(d))

- i 遍历每一路 retriever（BM25、vector，可能还有其它）
- rank_i(d) 是 d 在第 i 路里的排名（1-based）
- d 没出现在第 i 路 top-N 里就视为这一项 = 0
- k_rrf 是平滑常数，论文经验值 60；它的作用是抑制 rank-1 的过强影响：
    rank=1 -> 1/61 ≈ 0.0164
    rank=2 -> 1/62 ≈ 0.0161
  差距很小，所以 "两路都把 d 排在前几" > "一路排第 1 但另一路压根没出现"。

RRF 相比 weighted-sum 的好处：
- 不需要 score normalization——BM25 score 跟 cosine 不在一个量纲上，
  normalize 出来的结果跨 query 不稳定
- 几乎无参数，k_rrf 几乎不用动
- 对单路偶尔抽风很鲁棒：错排只贡献 1/(k_rrf+rank) 这点分

代价：丢掉了"绝对置信度"——cos=0.95 vs cos=0.6 在 RRF 眼里只差一个 rank。
要做 confidence threshold 之类的下游逻辑，得另算。
"""


DEFAULT_K_RRF = 60


def _default_key(chunk: dict) -> Hashable:
    """融合 key 用 chunk_id（chunking.py 保证一定存在且唯一）。"""
    return chunk["chunk_id"]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict]],
    k_rrf: int = DEFAULT_K_RRF,
    key: Callable[[dict], Hashable] = _default_key,
) -> list[dict]:
    """
    输入：多路 ranked list，每路是按相关度降序排好的 chunk dict 列表。
    输出：融合后按 RRF score 降序的 chunk dict 列表。
         返回字典里 'score' 字段被替换成 RRF score；
         同时塞一个 '_fusion' 字段记录每一路的 rank 和原始 score，debug 用。

    时间复杂度：O(R · N)，R = 路数，N = 每路 top-N 长度。
    """
    if k_rrf <= 0:
        raise ValueError(f"k_rrf must be positive, got {k_rrf}")

    # key -> 累积 RRF score
    fused_scores: dict[Hashable, float] = {}
    # key -> 第一次见到的 chunk dict（用来保留 payload）
    representative: dict[Hashable, dict] = {}
    # key -> 每一路的 (retriever_index, rank, original_score)，debug 用
    components: dict[Hashable, list[tuple[int, int, float]]] = {}

    for retriever_idx, ranked in enumerate(ranked_lists):
        for rank_zero_based, chunk in enumerate(ranked):
            rank = rank_zero_based + 1  # RRF 用 1-based rank
            kkey = key(chunk)

            fused_scores[kkey] = fused_scores.get(kkey, 0.0) + 1.0 / (k_rrf + rank)

            # 第一次见这个 key 就把 chunk payload 存下来
            # 后面再见到的同 key chunk（来自另一路）payload 应该是一样的，跳过即可
            if kkey not in representative:
                representative[kkey] = chunk

            components.setdefault(kkey, []).append(
                (retriever_idx, rank, float(chunk.get("score", 0.0)))
            )

    # 按 RRF score 降序排序
    fused = []
    for kkey, rrf_score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = dict(representative[kkey])  # 复制一份，不污染原始输入
        chunk["score"] = rrf_score
        chunk["_fusion"] = [
            {"retriever": ri, "rank": r, "score": s}
            for ri, r, s in components[kkey]
        ]
        fused.append(chunk)

    return fused


# ===== Example usage =====
# bm25_hits = bm25.top_k_retrieve(query, k=50)
# vector_hits = qdrant.top_k_retrieve(query, k=50)
# fused = reciprocal_rank_fusion([bm25_hits, vector_hits], k_rrf=60)
# top_5 = fused[:5]
