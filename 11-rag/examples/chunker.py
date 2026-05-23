"""
chunker.py — 文本分块

将长文档切分为固定大小的文本块，相邻块之间保留 overlap 字符的重叠，
防止关键信息落在块的边界处被切断。
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    按字符数切分文本，支持滑动窗口 overlap。

    Args:
        text:       原始文本字符串
        chunk_size: 每块的最大字符数，默认 512
        overlap:    相邻块的重叠字符数，默认 64

    Returns:
        文本块列表，去掉空白块

    Example:
        >>> chunks = chunk_text("A" * 1000, chunk_size=200, overlap=40)
        >>> len(chunks)  # ceil((1000 - 40) / (200 - 40)) = 7
        7
    """
    if not text or chunk_size <= 0:
        return []

    # overlap 不能大于等于 chunk_size，否则步长为 0 导致死循环
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) 必须小于 chunk_size ({chunk_size})")

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
