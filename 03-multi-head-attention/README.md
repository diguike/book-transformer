---
title: Multi-Head Attention
feishu_url: "https://www.feishu.cn/wiki/Oy3zwxitciaps4k9jRwcMt6nnBe"
last_synced: "2026-05-08T00:24:08"
---

## 单头 Attention 的局限

第 2 章实现的 Scaled Dot-Product Attention，本质上是在整个 `d_model` 维度空间里做一次"软查找"：给定一个 Query 向量，计算它与所有 Key 向量的相似度，然后对 Value 做加权求和。

这个过程只产生一种相似度度量。模型在同一时刻只能用一种方式去"关注"其他位置——要么偏向捕捉句法依赖，要么偏向捕捉语义相关，要么偏向指代关系，但没办法同时学多种模式。

举一个具体的例子。对于句子 "The animal didn't cross the street because it was too tired"，理解 "it" 指代 "animal" 涉及指代消解；而理解整句的语义需要关注 "cross"、"street"、"tired" 之间的搭配关系。这是两种性质不同的依赖，单次 Attention 很难同时表达好。

原始论文（Vaswani et al., 2017）给出的解法是：**把 `d_model` 维度切成 `h` 份，每一份独立做 Attention，最后把结果拼起来**。这就是 Multi-Head Attention。

## 多头的设计

Multi-Head Attention 的核心思路：与其在 512 维的完整空间里做一次 Attention，不如把 Q、K、V 分别投影到 `h` 个较小的子空间（每个子空间维度 `d_k = d_model / h`），在每个子空间里独立计算 Attention，最后把 `h` 个结果拼回来。

原始论文中，`d_model = 512`，`h = 8`，所以每个头的维度是 `d_k = 64`。

每个"头"有自己独立的投影矩阵 `W_i^Q`、`W_i^K`、`W_i^V`，维度都是 `(d_model, d_k)`。这意味着每个头会从不同的角度去看同一个输入。训练结束后，不同头自然分化出不同的"关注模式"——这是优化过程的涌现，而不是人为规定的。

![多头 Q、K、V 投影示意](assets/transformer_attention_heads_qkv.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

设计要点：

- `h` 个头并行计算，不是串行，训练时效率没有损失
- 每个头的 `d_k` 更小，单次 Attention 计算量反而下降
- 最终输出通过 `W^O`（维度 `(h * d_v, d_model)`）投影回原始维度

## 计算过程

Multi-Head Attention 分四步：

**第一步：线性投影**

对输入 Q、K、V，用 `h` 组不同的权重矩阵分别投影：

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

其中 `W_i^Q ∈ R^(d_model × d_k)`，`W_i^K ∈ R^(d_model × d_k)`，`W_i^V ∈ R^(d_model × d_v)`。

**第二步：分头计算 Attention**

每个头独立执行 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

输出维度 `(batch, seq_len, d_v)`。

**第三步：拼接（Concatenate）**

把 `h` 个头的输出在最后一个维度拼接：

```
MultiHead = Concat(head_1, head_2, ..., head_h)
```

拼接后维度为 `(batch, seq_len, h * d_v)`。

![多头输出拼接](assets/transformer_attention_heads_z.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

**第四步：输出投影**

用 `W^O ∈ R^(h*d_v × d_model)` 做一次线性变换，把维度映射回 `d_model`：

```
MultiHead = Concat(head_1, ..., head_h) W^O
```

![输出投影矩阵 W^O](assets/transformer_attention_heads_weight_matrix_o.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

整个计算流程如下：

```
输入 Q, K, V  (batch, seq_len, d_model)
      ↓
  [投影] × h 组 W^Q, W^K, W^V
      ↓
每个头: (batch, seq_len, d_k)
      ↓
  [Scaled Dot-Product Attention] × h
      ↓
每个头输出: (batch, seq_len, d_v)
      ↓
  [Concat 在 d_v 维度]
      ↓
拼接结果: (batch, seq_len, h * d_v)
      ↓
  [输出投影 W^O]
      ↓
最终输出: (batch, seq_len, d_model)
```

![Multi-Head Attention 完整流程总览](assets/transformer_multi-headed_self-attention-recap.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

实现时，通常不用真的创建 `h` 个独立的矩阵，而是用一个大矩阵 `W^Q ∈ R^(d_model × d_model)` 一次性投影，再 reshape 成 `(batch, h, seq_len, d_k)` 的形式，然后并行计算所有头的 Attention。这样实现更简洁，也更容易利用矩阵乘法的批处理能力。

## 参数量分析

一个容易误解的地方：Multi-Head Attention 的参数量和单头 Attention（如果单头使用相同维度投影）差不多，并不是"多了 h 倍"。

原因在于参数被"切分"了。以 `d_model=512, h=8` 为例：

| 矩阵 | 多头（h=8，每头 d_k=64） | 单头（d_k=512） |
|------|------------------------|----------------|
| `W^Q` | `512 × 512 = 262,144` | `512 × 512 = 262,144` |
| `W^K` | `512 × 512 = 262,144` | `512 × 512 = 262,144` |
| `W^V` | `512 × 512 = 262,144` | `512 × 512 = 262,144` |
| `W^O` | `512 × 512 = 262,144` | 无（或等效） |

多头方案多了一个 `W^O`，但实质上参数总量是可比的。多头的优势不在于参数量，而在于**用相同参数量表达了更丰富的特征**。

用 PyTorch 内置模块验证一下：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
total_params = sum(p.numel() for p in mha.parameters())
print(total_params)  # 1,049,600 ≈ 4 × 512 × 512 + 512 + 512
```

输出结果约为 104 万，正好是 4 个 `512×512` 矩阵加上 bias 的量。

## 不同头学到什么

不同头究竟学到什么，是一个可以实验验证的问题。

![不同注意力头关注的位置（示例一）](assets/transformer_self-attention_visualization_2.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![不同注意力头关注的位置（示例二）](assets/transformer_self-attention_visualization_3.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

Clark et al.（2019）在 BERT 上对注意力头做了系统分析，几个典型发现：

**句法头**：某些头倾向于关注句法上的直接依存关系（如主语-谓语、动词-宾语）。给一个句子，这类头的注意力权重分布和依存分析树的边高度吻合。

**指代头**：某些头会把代词（"it", "they"）的注意力集中到它所指代的名词上。

**位置头**：某些头几乎只关注相邻位置——前一个词或后一个词。这类头编码的是局部上下文信息。

**[CLS] 头**：在 BERT 这类模型里，某些头会把大量注意力分配给 `[CLS]` 或 `[SEP]` token，这类头通常被认为承担了"信息汇聚"的功能。

需要说明的是，这些"功能"是从大量样本的注意力分布中统计出来的规律，不是严格的结论。现实情况更加复杂：同一个头在不同输入上可能表现出不同的行为，且很多头的功能难以用语言简单描述。

直觉上理解：多头机制给了模型足够的自由度，让不同的子空间专注于不同的特征维度。在足够大的数据和足够长的训练之后，模型会自然地把这些"自由度"分配给有意义的特征——这是梯度下降的结果，不是设计的。

---

> 本章来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
