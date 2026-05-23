"""
embedder.py — Embedding 封装

封装 sentence-transformers 的 embedding 操作，对外暴露简洁接口。
模型实例在模块级懒加载（首次调用时初始化），整个进程复用同一实例。
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding 模型名称（首次运行自动下载，约 80MB）
MODEL_NAME = "all-MiniLM-L6-v2"

# 模块级单例，避免重复加载模型
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """懒加载模型，保证全局只初始化一次。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    批量编码文本列表，用于建立索引。

    Args:
        texts: 文本字符串列表

    Returns:
        shape=(n, 384) 的 numpy float32 array，每行是一个文本的向量
    """
    model = _get_model()
    return model.encode(texts, show_progress_bar=len(texts) > 10)


def embed_query(query: str) -> np.ndarray:
    """
    编码单条查询文本，用于检索时的向量查询。

    Args:
        query: 查询字符串

    Returns:
        shape=(384,) 的 numpy float32 array
    """
    model = _get_model()
    return model.encode(query)
