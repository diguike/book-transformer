---
title: Embedding 的工程用法
feishu_url: "https://www.feishu.cn/wiki/O54BwmIxni3tNpkFY0ecW08FnOb"
last_synced: "2026-05-07T23:07:06"
---

## Token Embedding vs Sentence Embedding

在 Transformer 模型里，embedding 这个词在两个完全不同的场景下被使用，混淆它们会让很多工程实践走弯路。

**Token Embedding** 是模型内部的基础构件。输入文本经过分词后变成一串 token ID，embedding 层把每个 ID 映射成一个固定维度的向量。以 BERT-base 为例，词表大小约 30000，embedding 维度 768，所以 embedding 矩阵是一个 30000×768 的参数矩阵。每个 token 在序列中都有独立的向量表示，一句话包含多少个 token，就产生多少个 768 维向量。

这些向量经过后续的 attention 层后，每个位置的表示会融合上下文信息——"苹果"在"我吃了苹果"和"苹果发布了新手机"中，最终的向量表示是不同的。这也是 Transformer 相比静态词向量（Word2Vec）的核心优势。

**Sentence Embedding** 是把整句话压缩成一个向量。它解决的是不同场景下的工程需求：语义搜索要比较查询和文档的相关性，推荐系统要计算内容相似度，这些场景需要的是"一句话 = 一个向量"，而不是"一句话 = N 个向量"。

从 Token Embedding 到 Sentence Embedding 需要一个聚合步骤。常见做法有两种：取 `[CLS]` token 的向量（BERT 原始设计），或对所有 token 的向量取均值（mean pooling）。实践中 mean pooling 通常效果更好，因为它用到了序列里所有 token 的信息。

工程上实际用到的大多是 Sentence Embedding。Token Embedding 更多是模型内部机制，只有在做 token 级别任务（命名实体识别、词性标注等）时才会直接使用最后一层的 token 向量。

## 怎么得到 Sentence Embedding

### Mean Pooling vs [CLS] Token

`[CLS]` token 是 BERT 引入的特殊标记，位于每句话的开头。原始论文的设计意图是让这个 token 在经过 attention 层后汇聚整句的语义信息，用于下游分类任务。但多项研究发现，直接用 `[CLS]` 向量做句子相似度计算效果并不理想，各向量空间分布不均匀，余弦相似度的区分度差。

Mean pooling 对最后一层所有 token 的隐藏状态取平均，实验证明在语义相似度任务上优于 `[CLS]`。`sentence-transformers` 库默认使用 mean pooling，这也是目前最常用的方案。

需要注意的是，批量处理时序列会被 padding 到相同长度，padding token 不携带真实语义，均值计算时必须用 `attention_mask` 排除这些位置——只对 mask 为 1 的 token 取均值，而不是对整段序列（含 padding）直接平均。`sentence-transformers` 的 `encode()` 内部已处理这个细节，直接用 pipeline 或 `pipeline("feature-extraction")` 做简单平均则不会自动排除 padding。

### sentence-transformers 的用法

`sentence-transformers` 是目前工程上最成熟的 sentence embedding 库，封装了模型加载、pooling、归一化等细节：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# encode() 内部做了 tokenize → forward → mean pooling → normalize
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
```

`all-MiniLM-L6-v2` 是一个常用的轻量模型，输出维度 384，模型大小约 22MB，推理速度快，适合本地开发和对延迟敏感的场景。需要更高精度时可以换 `all-mpnet-base-v2`（维度 768，效果更好但更慢）。

`encode()` 默认返回 numpy 数组，传入 `convert_to_tensor=True` 可以得到 PyTorch tensor，便于在 GPU 上做后续计算。

## 余弦相似度

### 公式

两个向量 $\mathbf{a}$ 和 $\mathbf{b}$ 的余弦相似度定义为：

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

结果范围是 $[-1, 1]$，1 表示方向完全相同（语义高度相似），0 表示正交（无关），-1 表示方向相反。

### 为什么不用欧氏距离

欧氏距离衡量的是向量端点之间的绝对距离，对向量长度敏感。两个语义相同但表达长度不同的句子，其 embedding 的模长可能相差较大，导致欧氏距离偏大，即使归一化后也可能引入额外误差。

余弦相似度只关注向量方向，忽略模长，更适合衡量语义的相对距离。实际上，如果向量已经做过 L2 归一化（模长为 1），余弦相似度就等价于点积，可以用矩阵乘法高效计算。`sentence-transformers` 的 `encode()` 默认开启归一化（`normalize_embeddings=True`），所以直接用点积即可。

### 代码实现

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["今天天气很好", "阳光明媚", "他去了体育馆"]
embeddings = model.encode(sentences, normalize_embeddings=True)

# 已归一化，点积 == 余弦相似度
similarity_matrix = np.dot(embeddings, embeddings.T)
print(similarity_matrix)
```

`sentence-transformers` 也内置了工具函数 `util.cos_sim()`，接受 tensor 或 numpy 数组，返回相似度矩阵，处理批量计算更方便。

## 向量数据库

### 为什么需要向量数据库

假设有 100 万条文档，每条文档的 embedding 维度是 768。暴力搜索（brute force）需要计算查询向量与全部 100 万条的余弦相似度，即 100 万次向量点积，延迟在毫秒级别对于实时应用不可接受，且随数据量线性增长。

向量数据库的核心是 **ANN（Approximate Nearest Neighbor，近似最近邻）** 索引，用少量精度损失换取数量级的速度提升。

### ANN 索引原理简介

主流的 ANN 算法包括：

**HNSW（Hierarchical Navigable Small World）**：构建多层图结构，高层是稀疏的跳表式连接，低层是密集的近邻连接。查询时从高层开始，快速定位大致区域，再在低层精细搜索。查询复杂度约 $O(\log n)$，精度高，是目前工程上最常用的算法，Qdrant 和 Chroma 都默认使用 HNSW。

**IVF（Inverted File Index）**：先用 k-means 把向量空间划分成若干 cluster，查询时只在最近的几个 cluster 里搜索。适合超大规模数据集，内存占用比 HNSW 小，但精度稍低。Faiss 常用这种方式。

**LSH（Locality Sensitive Hashing）**：通过哈希函数把相似向量映射到相同的桶里，速度快但精度相对较低，目前在主流向量数据库中使用较少。

### Qdrant vs Chroma 选型对比

| 对比维度 | Qdrant | Chroma |
|---------|--------|--------|
| 定位 | 生产级向量数据库 | 本地开发/原型 |
| 部署方式 | Docker / 云服务 | 本地嵌入式或服务端 |
| 持久化 | 默认持久化到磁盘 | 默认内存，可持久化 |
| 过滤查询 | 支持复杂 payload 过滤 | 支持基础 metadata 过滤 |
| 性能 | 高并发场景表现好 | 适合小规模数据 |
| Python API | 官方 client | 原生 Python |
| 适用场景 | 生产部署、百万级以上数据 | 本地测试、RAG 原型开发 |

选型建议：原型阶段用 Chroma，开箱即用，零配置。上生产或数据量超过 10 万条，迁移到 Qdrant，稳定性和性能更有保障。两者的 Python API 设计类似，迁移成本不高。

## 工程实践要点

### Embedding 维度的权衡

维度越高，表达能力越强，但存储和计算开销也更大。100 万条向量，维度 384 需要约 1.5GB 存储（float32），维度 1536（OpenAI text-embedding-3-large）则需要 6GB。

在实际项目中，先用轻量模型（384 维）跑通流程，评估效果，确实需要更高精度再升级模型。不要一开始就上最大维度。

### 模型选择

几个常用的开源模型：

- `all-MiniLM-L6-v2`：384 维，22MB，推理快，适合资源受限场景
- `all-mpnet-base-v2`：768 维，420MB，精度更高，SBERT 基准上表现优秀
- `paraphrase-multilingual-MiniLM-L12-v2`：384 维，支持 50+ 语言，中文场景首选
- `BAAI/bge-large-zh-v1.5`：1024 维，专为中文优化，MTEB 中文排行榜靠前

中文业务场景推荐 `bge-large-zh-v1.5` 或 `paraphrase-multilingual-MiniLM-L12-v2`，不要用英文模型处理中文文本。

### 归一化

入库前对所有向量做 L2 归一化，好处有两点：

1. 余弦相似度可以直接用点积计算，速度更快
2. 向量数据库的距离度量设置为 `dot_product`（点积），比 cosine 计算路径更短

`sentence-transformers` 的 `encode(normalize_embeddings=True)` 会自动处理，不需要手动归一化。

### 批量推理性能

单条逐个调用 `encode()` 效率很低，批量传入列表可以充分利用 GPU 并行：

```python
# 低效：逐条推理
for sentence in sentences:
    emb = model.encode(sentence)

# 高效：批量推理
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
```

`batch_size` 的合理取值取决于显存和模型大小。`all-MiniLM-L6-v2` 在普通 GPU 上 batch_size=256 通常没问题。CPU 推理建议 batch_size=32，平衡内存和速度。

增量更新文档索引时，用 `collection.upsert()` 而不是 `collection.add()`——upsert 会自动跳过已存在的文档，只为新增或修改的文档重新计算 embedding 并更新索引，不需要全量重建。
