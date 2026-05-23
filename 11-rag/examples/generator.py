"""
generator.py — LLM 调用与回答生成

封装 OpenAI 兼容接口的调用，构造 RAG prompt 并生成回答。
支持 OpenAI API 和任何兼容 OpenAI 格式的本地服务（如 Ollama）。

环境变量配置：
    OPENAI_BASE_URL  默认 https://api.openai.com/v1
    OPENAI_API_KEY   必填
    OPENAI_MODEL     默认 gpt-3.5-turbo
"""

from __future__ import annotations

import os

from openai import OpenAI

# 从环境变量读取配置，方便切换 OpenAI / Ollama / 其他兼容服务
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量，例如：export OPENAI_API_KEY=sk-xxx")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# system prompt：明确限制 LLM 只能基于提供的上下文回答，不得编造
SYSTEM_PROMPT = """你是一个专业的文档问答助手。

规则：
1. 只根据下面用户提供的【上下文内容】回答问题，不要使用训练数据中的其他知识。
2. 如果上下文中没有足够的信息回答问题，直接回答："根据提供的文档，无法回答该问题。"
3. 回答要简洁准确，不要重复上下文原文，用自己的语言归纳。
4. 不要编造数据、引用、日期或任何上下文中未提及的信息。"""


def _build_user_prompt(question: str, contexts: list[dict]) -> str:
    """
    将检索到的文本块和用户问题拼装成 user prompt。

    Args:
        question: 用户问题
        contexts: retriever.retrieve() 返回的字典列表

    Returns:
        格式化的 user prompt 字符串
    """
    # 将各文本块拼接为上下文段落，用分隔线隔开
    context_parts = []
    for i, ctx in enumerate(contexts, start=1):
        context_parts.append(f"[{i}]\n{ctx['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    return f"上下文内容：\n\n{context_str}\n\n---\n\n问题：{question}"


def generate_answer(question: str, contexts: list[dict]) -> str:
    """
    基于检索到的上下文，调用 LLM 生成回答。

    Args:
        question: 用户问题字符串
        contexts: retriever.retrieve() 返回的字典列表，每个包含 'text' 字段

    Returns:
        LLM 生成的回答字符串
    """
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    user_prompt = _build_user_prompt(question, contexts)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,   # 低温度：减少随机性，让回答更稳定、更忠实于原文
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


def generate_answer_stream(question: str, contexts: list[dict]):
    """流式生成回答，逐块 yield 文本片段（生产环境推荐用这种方式）"""
    context_text = "\n---\n".join(c["text"] for c in contexts)
    messages = [
        {
            "role": "system",
            "content": "你是一个问答助手。只根据下面提供的上下文内容回答问题。"
                       "如果上下文中没有相关信息，回答"根据提供的文档，无法回答该问题"。"
                       "不要编造信息。",
        },
        {
            "role": "user",
            "content": f"上下文：\n---\n{context_text}\n---\n\n问题：{question}",
        },
    ]
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,  # 开启流式
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
