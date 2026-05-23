"""
searcher.py — 执行语义搜索

将用户查询编码为向量，在 ChromaDB 中执行近似最近邻搜索，返回 top-K 结果。
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer


def search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """
    对查询做 embedding，在向量库中检索最相关的文档。

    Args:
        query:      用户查询字符串
        collection: ChromaDB collection 对象
        model:      已加载的 SentenceTransformer 模型
        top_k:      返回结果数量，默认 5

    Returns:
        字典列表，每个字典包含：
          - document: 文档原文（str）
          - score:    相似度分数，范围 [0, 1]，越高越相关（float）
          - id:       文档 ID（str）
    """
    # 编码查询，返回 shape=(384,) 的 numpy array
    query_embedding = model.encode(query)

    # ChromaDB 的 query() 支持批量查询，参数和返回值都是列表的列表
    # 这里只查询单条，取 [0] 获取第一条查询的结果
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "distances"],
    )

    # 将结果拼装成易用的字典列表
    output = []
    documents = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc, dist, doc_id in zip(documents, distances, ids):
        # ChromaDB 余弦空间返回的是距离（distance = 1 - similarity）
        # 转换回相似度分数，并 clip 到 [0, 1] 防止浮点误差
        score = max(0.0, min(1.0, 1.0 - dist))
        output.append({
            "document": doc,
            "score": score,
            "id": doc_id,
        })

    return output
