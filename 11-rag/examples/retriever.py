"""
retriever.py — 向量存储与检索

用 ChromaDB embedded 模式做持久化向量存储，
提供文档块索引和语义检索能力。
"""

from __future__ import annotations

import chromadb

import embedder

# ChromaDB 持久化路径
CHROMA_PATH = "./chroma_db"
# collection 名称
COLLECTION_NAME = "rag_chunks"

# 模块级 ChromaDB 客户端（懒加载）
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """获取（或创建）ChromaDB collection，保证全局复用同一实例。"""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_chunks(chunks: list[str], source_file: str) -> None:
    """
    对文本块列表做 embedding 并写入 ChromaDB。

    Args:
        chunks:      分块后的文本列表
        source_file: 来源文件名，存入 metadata 用于展示引用来源
    """
    collection = _get_collection()

    # 批量编码所有文本块
    embeddings = embedder.embed_texts(chunks)

    # 构造 ID：来源文件名 + 块序号，确保同一文档重新索引时能正确覆盖
    ids = [f"{source_file}_{i}" for i in range(len(chunks))]

    # metadata 存储来源信息，检索结果中可直接读取
    metadatas = [
        {"source": source_file, "chunk_id": i}
        for i in range(len(chunks))
    ]

    # upsert：ID 已存在时更新，不存在时插入
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
    )


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    检索与查询最相关的文本块。

    Args:
        query:  用户查询字符串
        top_k:  返回结果数量，默认 4

    Returns:
        字典列表，每个字典包含：
          - text:     文本块原文（str）
          - score:    相似度分数，范围 [0, 1]，越高越相关（float）
          - source:   来源文件名（str）
          - chunk_id: 块在文档中的序号（int）
    """
    collection = _get_collection()

    # 编码查询
    query_vector = embedder.embed_query(query)

    # 执行向量搜索
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    # 解包嵌套列表（ChromaDB 支持批量查询，返回结果是列表的列表）
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    output = []
    for doc, dist, meta in zip(docs, distances, metas):
        # 余弦距离转相似度，clip 到 [0, 1]
        score = max(0.0, min(1.0, 1.0 - dist))
        output.append({
            "text": doc,
            "score": score,
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", -1),
        })

    return output


def collection_count() -> int:
    """返回当前 collection 中的文档块数量。"""
    return _get_collection().count()
