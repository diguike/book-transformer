"""
ChromaDB 本地向量库演示

运行环境：CPU 即可，无需 GPU
依赖安装：pip install -r requirements.txt

演示内容：
  1. 创建 ChromaDB 本地向量库
  2. 插入 10 条文档（含 embedding）
  3. 做语义检索，返回 top-3 最相关文档

ChromaDB 核心概念：
  - Client：数据库连接，管理所有 collection
  - Collection：类似关系型数据库的"表"，存储一组向量及其关联文档
  - Document：原始文本内容，与 embedding 一一对应
  - Embedding：文档的向量表示，存储在 HNSW 索引中供 ANN 搜索
  - Metadata：附加在文档上的键值对，支持过滤查询
"""

import chromadb
from sentence_transformers import SentenceTransformer

# 初始化 embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 ChromaDB 客户端（持久化到本地目录 ./chroma_db）
# 使用 PersistentClient 确保数据在进程重启后不丢失
# 开发阶段也可以用 chromadb.Client() 只存在内存里
client = chromadb.PersistentClient(path="./chroma_db")

# 创建（或获取已有的）collection
# collection 相当于一个独立的向量表，每个 collection 有独立的 HNSW 索引
# get_or_create_collection：已存在则获取，不存在则新建，幂等操作
collection = client.get_or_create_collection(
    name="tech_docs",
    # metadata 可以指定距离度量方式：cosine / l2 / ip（内积）
    metadata={"hnsw:space": "cosine"}
)

# 准备 10 条示例文档（技术文档场景）
documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "PyTorch is a deep learning framework developed by Meta AI Research.",
    "Transformers are neural networks that use self-attention to process sequential data.",
    "BERT is a bidirectional transformer model pre-trained on large text corpora.",
    "GPT models generate text by predicting the next token in a sequence.",
    "Docker is a platform for building and running containerized applications.",
    "Kubernetes orchestrates containers across clusters of machines.",
    "PostgreSQL is a powerful open-source relational database system.",
    "Redis is an in-memory data structure store used as a cache and message broker.",
    "Nginx is a high-performance web server and reverse proxy.",
]

# 每条文档的 metadata，可用于后续过滤
metadatas = [
    {"category": "programming_language", "difficulty": "beginner"},
    {"category": "ml_framework",         "difficulty": "intermediate"},
    {"category": "ml_concept",           "difficulty": "intermediate"},
    {"category": "ml_model",             "difficulty": "advanced"},
    {"category": "ml_model",             "difficulty": "advanced"},
    {"category": "devops",               "difficulty": "beginner"},
    {"category": "devops",               "difficulty": "intermediate"},
    {"category": "database",             "difficulty": "intermediate"},
    {"category": "database",             "difficulty": "beginner"},
    {"category": "infrastructure",       "difficulty": "intermediate"},
]

# 文档 ID，在 collection 内必须唯一
doc_ids = [f"doc_{i}" for i in range(len(documents))]

# 提前计算所有文档的 embedding
# 实际项目中这一步在数据入库时执行一次，后续查询不需要重算
print("正在计算文档 embedding...")
doc_embeddings = model.encode(documents, normalize_embeddings=True).tolist()

# 将文档、embedding、metadata 批量写入 collection
# ChromaDB 的 upsert：已存在相同 ID 则更新，不存在则插入
collection.upsert(
    ids=doc_ids,
    documents=documents,
    embeddings=doc_embeddings,
    metadatas=metadatas,
)

print(f"已插入 {collection.count()} 条文档")
print()

# ── 语义检索 ──────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 3, category_filter: str = None):
    """
    对 collection 做语义检索，返回 top_k 最相关文档。

    Args:
        query: 查询文本
        top_k: 返回结果数量
        category_filter: 可选，按 metadata 的 category 字段过滤
    """
    # 计算查询向量
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # 构建过滤条件（可选）
    where_filter = None
    if category_filter:
        # ChromaDB 的 where 语法：字段名 + 操作符
        where_filter = {"category": {"$eq": category_filter}}

    # 执行 ANN 搜索
    # ChromaDB 内部使用 HNSW 索引，在百万级数据量下仍能快速返回结果
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return results


# 查询 1：无过滤条件，全库语义检索
print("=" * 60)
query1 = "how do neural networks learn from text data?"
print(f"查询：{query1}")
print()

results1 = search(query1, top_k=3)
for i, (doc, meta, dist) in enumerate(zip(
    results1["documents"][0],
    results1["metadatas"][0],
    results1["distances"][0],
)):
    # ChromaDB cosine 距离 = 1 - cosine 相似度，越小越相关
    similarity = 1 - dist
    print(f"  Top-{i+1}（相似度 {similarity:.4f}，类别 {meta['category']}）：")
    print(f"    {doc}")
    print()

# 查询 2：带 category 过滤，只在 devops 类别里搜索
print("=" * 60)
query2 = "container management and deployment"
print(f"查询：{query2}（限定类别：devops）")
print()

results2 = search(query2, top_k=3, category_filter="devops")
for i, (doc, meta, dist) in enumerate(zip(
    results2["documents"][0],
    results2["metadatas"][0],
    results2["distances"][0],
)):
    similarity = 1 - dist
    print(f"  Top-{i+1}（相似度 {similarity:.4f}）：")
    print(f"    {doc}")
    print()

# 查询 3：数据库相关
print("=" * 60)
query3 = "fast data storage for caching"
print(f"查询：{query3}")
print()

results3 = search(query3, top_k=3)
for i, (doc, meta, dist) in enumerate(zip(
    results3["documents"][0],
    results3["metadatas"][0],
    results3["distances"][0],
)):
    similarity = 1 - dist
    print(f"  Top-{i+1}（相似度 {similarity:.4f}，类别 {meta['category']}）：")
    print(f"    {doc}")
    print()

print("=" * 60)
print("ChromaDB 本地持久化目录：./chroma_db")
print("重启后数据仍然存在，无需重新插入。")
