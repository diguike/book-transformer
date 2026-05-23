---
title: 从文字到向量
feishu_url: "https://www.feishu.cn/wiki/SVcGwFcNhigAZ4kNF7DcbxIInWe"
last_synced: "2026-05-07T23:06:23"
---

## 计算机无法直接处理文字

计算机只认识数字。无论是排序、加法还是矩阵乘法，底层操作的都是数值。文字本身不是数字，所以在把文本送进任何机器学习模型之前，必须先把它转换成数值表示。

这个问题早期有一个直接但粗糙的解法：one-hot 编码。假设词表有 50000 个词，每个词用一个 50000 维的向量表示，向量里只有对应位置为 1，其余全是 0。

![one-hot 编码示意](assets/one-hot-vocabulary-example.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

这个方案有两个根本性的缺陷：

1. **维度爆炸**。词表越大，向量越稀疏，内存和计算开销都不可接受。
2. **没有语义信息**。"猫"和"狗"的 one-hot 向量之间的距离，和"猫"与"汽车"完全相同——模型无法从向量本身感知词语之间的关系。

Transformer 解决这个问题的路径分三步：Tokenization → Token ID → Embedding。每一步都值得单独说清楚。

## Tokenization：文本的切分方式

Tokenization 是把原始文本切分成模型可以处理的最小单元（token）的过程。最朴素的做法是按词切分——空格隔开的就是一个词。但这有明显的问题：

- **未登录词**（OOV）：训练时没见过的词，推理时无法处理。
- **形态变化**：英语中 `run`、`runs`、`running` 是同一个词根，简单按词切分会当成三个无关词。
- **多语言**：中文、日文没有空格，按字切又太细。

目前主流方案是 **BPE（Byte Pair Encoding）**，BERT、GPT 系列都在用它的变体。

### BPE 的基本原理

BPE 的思路是从字符级别出发，反复合并出现频率最高的相邻字节对，直到达到预设的词表大小。

以一个极简例子说明。假设训练语料里只有这些词（括号内是频次）：

```
low (5), lower (2), newest (6), widest (3)
```

初始化时，把每个词拆成字符序列，并加上单词结束符 `</w>`：

```
l o w </w>          (5)
l o w e r </w>      (2)
n e w e s t </w>    (6)
w i d e s t </w>    (3)
```

统计所有相邻字节对的频次：`e s` 出现了 9 次（6+3），是频率最高的。合并后：

```
l o w </w>          (5)
l o w e r </w>      (2)
n e w es t </w>     (6)
w i d es t </w>     (3)
```

继续合并 `es t`（9 次）、`est </w>`（9 次），最终词表里会出现 `est</w>` 这个子词单元。`newest` 和 `widest` 共享了这个后缀，模型可以学习到它们在形态上的关联。

**BPE 的工程价值在于**：词表大小可控（一般 30k-50k），既不会因词太多造成稀疏问题，也不会因切得太细丢失语义单元。遇到未登录词时，会退化成更细粒度的子词组合，不会直接报错。

### 切分方式影响模型性能

Tokenizer 的选择直接影响模型能处理的序列长度和语义捕获能力。一个中文字符在 BERT 的 tokenizer 里通常是一个 token，但在某些基于 BPE 的 tokenizer 里可能被切成多个字节级别的 token，使得同样的语料消耗更多序列长度，也更难学习汉字级别的语义。

GPT-4 使用的 `cl100k_base` tokenizer，对中文的压缩比远好于 GPT-2 时代的 tokenizer，这直接降低了处理相同中文文本的 token 数，减少了推理成本。

## Token ID：从字符到整数

Tokenizer 切分完成后，每个 token 需要映射到一个整数 ID，这就是词表（vocabulary）的作用。词表是一个固定的映射表：

```
"hello" → 7592
"world" → 2088
"[CLS]" → 101
"[SEP]" → 102
```

这个映射是训练前就确定的，推理时不会改变。

![词表与 Token ID 映射](assets/vocabulary.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

BERT 的词表大小是 30522，每个 token 对应 0-30521 之间的一个整数。GPT-2 是 50257，GPT-4 的 `cl100k_base` 是 100277。

还有一类特殊 token，由模型设计者规定含义：

- `[CLS]`：BERT 用它标记序列开头，其对应的输出向量被用于分类任务。
- `[SEP]`：分隔两个句子，或标记序列结尾。
- `[PAD]`：填充短序列，使 batch 内所有序列等长。
- `[MASK]`：BERT 预训练时随机遮盖 token 用于预测。

这些特殊 token 不是自然语言，但在模型的计算流程中承担特定角色。

## Embedding：从整数到向量

拿到 Token ID 之后，模型还不能直接用它做计算。整数 ID 只是索引，没有携带任何语义信息——ID 7592 和 7593 之间的差值 1，不代表这两个词的含义相近。

**Embedding 层**解决这个问题。它本质上是一个查找表（lookup table）：

```
embedding_matrix: shape [vocab_size, d_model]
```

![Embedding lookup 示意](assets/embeddings.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

给定 Token ID，就从这个矩阵里取出对应行，得到一个 `d_model` 维的稠密向量。`d_model` 在 BERT-base 里是 768，在 GPT-3 里是 12288。

这个矩阵的初始值是随机的，但会在训练过程中通过反向传播不断调整。训练结束后，语义相近的词会自然地聚集到向量空间的相近区域——这是学出来的，不是人工设计的。

### 为什么不用 one-hot

one-hot 向量的维度等于词表大小（几万到几十万），而 embedding 向量的维度只有几百到几千。

更重要的是密度：one-hot 是极度稀疏的，两个词的内积永远是 0（除非是同一个词）。embedding 是稠密向量，可以通过点积或余弦相似度度量两个词的语义距离。

### 维度的含义

embedding 的每个维度没有明确的人工定义的含义。训练结束后，某些维度可能隐约对应"是否是名词"、"情感正负"、"领域归属"，但这是涌现出来的，不是预先规定的。

维度越高，可以编码的信息越丰富，但计算开销也越大。BERT-base 用 768 维，是在表达能力和效率之间取的工程平衡点。

## 向量空间里的语义关系

embedding 最有名的演示是：

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
```

这个等式的成立需要一个前提：在训练语料里，"king"和"man"的共现模式，与"queen"和"woman"的共现模式高度类似。模型在训练中学到了"皇室地位"和"性别"是两个独立的语义方向，并在向量空间中用不同的维度方向编码了它们。

这种线性结构并不是设计出来的，而是大规模语料训练的副产品。它说明高维向量空间可以同时编码多个语义维度，向量运算在某种程度上对应语义操作。

在工程上，这意味着可以用向量距离（余弦相似度、点积）来衡量语义相近程度，这是语义搜索、推荐系统、RAG 等应用的基础。

## 工程意义：embedding 是 Transformer 的输入层

把上面三步串起来：

```
原始文本
  ↓ Tokenization
["hello", "world"]
  ↓ Token ID
[7592, 2088]
  ↓ Embedding lookup
[[0.12, -0.34, 0.56, ...],   # shape: [d_model]
 [0.78,  0.11, -0.23, ...]]  # shape: [d_model]
  ↓
shape: [seq_len, d_model]    # Transformer 的输入
```

Transformer 的输入是一个 `[seq_len, d_model]` 的矩阵。`seq_len` 是 token 数量，`d_model` 是 embedding 维度。后续所有的 Attention 计算都在这个矩阵上进行。

embedding 层是 Transformer 和自然语言之间的唯一接口。理解了这一层，后面的 Attention 机制才有了操作的对象。

---

代码示例在 `examples/` 目录下，需要先安装依赖：

```bash
pip install -r examples/requirements.txt
```

- `01_tokenizer_demo.py`：演示 BERT tokenizer 的使用，展示 token ids 和特殊 token
- `02_embedding_demo.py`：提取 BERT embedding，计算两个句子的余弦相似度

---

> 本章来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
