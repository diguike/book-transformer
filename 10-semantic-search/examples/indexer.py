"""
indexer.py — 构建和加载向量索引

负责将文档列表编码为向量并持久化到 ChromaDB。
ChromaDB 使用 embedded 模式（无需独立服务进程），数据存储在本地 ./chroma_db 目录。
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

# 向量库持久化路径
CHROMA_PATH = "./chroma_db"
# ChromaDB collection 名称
COLLECTION_NAME = "documents"
# Embedding 模型名称（首次运行自动下载，约 80MB）
MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(documents: list[str]) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    对文档列表做 embedding，构建并持久化向量索引。

    Args:
        documents: 文档字符串列表

    Returns:
        (collection, model) — ChromaDB collection 和已加载的模型
    """
    # 加载 Embedding 模型
    model = SentenceTransformer(MODEL_NAME)

    # 初始化持久化 ChromaDB 客户端
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 获取或创建 collection，指定余弦距离度量
    # 余弦距离 = 1 - 余弦相似度，取值 [0, 2]，越小越相似
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 批量编码文档，返回 shape=(n_docs, 384) 的 numpy array
    embeddings = model.encode(documents, show_progress_bar=True)

    # 构造文档 ID（ChromaDB 要求 ID 为字符串）
    ids = [str(i) for i in range(len(documents))]

    # upsert 比 add 更安全：ID 已存在时更新，不存在时插入，避免重复运行报错
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
    )

    return collection, model


def load_index() -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    加载已有的向量索引。调用前需确保 ./chroma_db 目录存在。

    Returns:
        (collection, model) — ChromaDB collection 和已加载的模型
    """
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    return collection, model
