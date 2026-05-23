---
title: 三种架构变体
feishu_url: "https://www.feishu.cn/wiki/QQ2BwygVIipDB3kk90dcBqNXnqe"
last_synced: "2026-05-07T23:06:52"
---

## 从原始 Transformer 到三条路线

2017 年的原始 Transformer 是一个完整的 Encoder-Decoder 结构，专为机器翻译设计。Encoder 把源语言句子编码成一组向量，Decoder 逐词生成目标语言。这个设计在翻译任务上效果显著，但研究者很快意识到：并不是所有任务都需要这两个模块。

2018 年，两条新路线几乎同时出现：Google 发布 BERT，只用 Encoder；OpenAI 发布 GPT，只用 Decoder。这并不是偶然的分叉，而是对"什么任务需要什么能力"的工程判断。

**核心区别在注意力机制的方向性：**

- Encoder 使用**双向注意力**：每个 token 可以看到序列中所有其他 token（包括左边和右边的）。这对于理解任务非常有用——判断情感、识别实体时，需要结合上下文才能做出准确判断。
- Decoder 使用**因果注意力（Causal Attention）**：每个 token 只能看到它左边的 token，不能看到未来的内容。这是生成任务的自然约束——生成第 5 个词时，你只知道前 4 个词。
- Encoder-Decoder 结合两者：Encoder 双向理解输入，Decoder 基于 Encoder 的输出自回归地生成输出。

下面分三条路线详细展开。

## Encoder-Only：BERT 系列

### 结构与注意力

BERT（Bidirectional Encoder Representations from Transformers）只使用 Transformer 的 Encoder 部分，堆叠多层。以 `bert-base-uncased` 为例：12 层 Encoder，768 维隐层，12 个注意力头，参数量 110M。

每一层的自注意力是**全双向的**：序列中任意两个位置都可以互相关注。对于句子"The bank can guarantee deposits"，模型在处理"bank"时，同时看到"deposits"，从而正确判断这里的"bank"是金融机构而非河岸。

### 预训练任务：MLM

BERT 的预训练使用**掩码语言模型（Masked Language Model, MLM）**：随机遮盖输入中 15% 的 token，让模型预测被遮盖的内容。

```
输入：The [MASK] can guarantee deposits
目标：bank
```

这个任务天然要求模型同时利用左右上下文，所以需要双向注意力。

BERT 还有第二个预训练任务——下句预测（Next Sentence Prediction, NSP），但后续研究（RoBERTa）表明 NSP 对下游任务帮助不大，可以去掉。

### 适合的任务

Encoder-Only 的输出是每个 token 的向量表示，或者整个句子的向量表示（通常用 `[CLS]` token 的向量）。这些向量经过微调后用于：

- **文本分类**：情感分析、意图识别、主题分类——在 `[CLS]` 向量上接一个线性层
- **命名实体识别（NER）**：序列标注，每个 token 预测一个标签（B-PER、I-ORG 等）
- **文本相似度 / 语义匹配**：双句子输入，判断是否语义相近（用于搜索、去重）
- **阅读理解 / 抽取式问答**：定位答案在段落中的起止位置

BERT 系列**不适合生成任务**——因为它的结构不支持自回归解码。

### 主要变种

| 模型 | 改进点 |
|------|--------|
| RoBERTa | 更多数据、更长训练、去掉 NSP、动态 masking |
| ALBERT | 参数共享 + 嵌入分解，大幅压缩参数量 |
| DistilBERT | 知识蒸馏，参数量减少 40%，速度提升 60% |
| DeBERTa | 分离位置编码与内容编码，引入解耦注意力 |
| MacBERT | 中文 BERT，用相似词替代 [MASK] 减小预训练与微调的差异 |

## Decoder-Only：GPT 系列

### 结构与注意力

GPT（Generative Pre-trained Transformer）只使用 Transformer 的 Decoder 部分，但去掉了原始 Decoder 中的 Cross-Attention（因为没有 Encoder 的输出可以关注）。每层只有自注意力 + FFN。

关键约束是**因果掩码（Causal Mask）**：在注意力矩阵上施加一个上三角掩码，把未来位置的注意力权重设为负无穷，经过 Softmax 后变为 0。这样 position i 的 token 只能看到 position 0 到 i 的内容。

```
位置 0 1 2 3
位置 0: ✓ ✗ ✗ ✗
位置 1: ✓ ✓ ✗ ✗
位置 2: ✓ ✓ ✓ ✗
位置 3: ✓ ✓ ✓ ✓
```

### 预训练任务：CLM

GPT 的预训练任务是**因果语言模型（Causal Language Model, CLM）**，也叫下一词预测：给定前 n 个 token，预测第 n+1 个。

```
输入：The cat sat on the
目标：mat
```

对于长度为 L 的序列，一次前向传播可以同时计算 L 个预测任务（每个位置预测下一个词），训练非常高效。

### 为什么现代大模型都是 Decoder-Only

这是一个值得认真回答的问题。从 GPT-3 开始，几乎所有大规模语言模型（LLaMA、Mistral、Qwen、GLM 的生成版本等）都选择了 Decoder-Only 架构，原因是多方面的：

**1. Few-shot 涌现能力**

GPT-3 的论文（2020）发现，当模型规模足够大时，Decoder-Only 模型在没有微调的情况下，仅通过 few-shot prompt 就能完成多种任务。这种能力被称为 In-Context Learning（ICL），在 Encoder-Only 模型上没有被观察到。

**2. 统一的训练目标**

CLM 是一个极其简单的自监督目标，不需要任何标注数据，直接用所有文本都能训练。随着训练数据和参数规模增大，模型能力持续提升，没有明显的天花板。

**3. 生成是最通用的接口**

分类、问答、翻译、摘要……理论上都可以被表述为"给定 prompt，生成答案"。Decoder-Only 天然支持这种统一范式，而 Encoder-Only 在没有额外结构的情况下无法生成。

**4. KV Cache 推理效率**

自回归解码时，每一步只需要计算新 token 的 Key 和 Value，之前的可以缓存复用（KV Cache）。这让 Decoder-Only 模型的推理可以高效实现，是工程落地的关键。

### 主要变种

| 模型 | 特点 |
|------|------|
| GPT-2 | 1.5B，开源，可本地运行 |
| GPT-3 / GPT-4 | API 访问，闭源 |
| LLaMA 2 / 3 | Meta 开源，工业级可用 |
| Mistral / Mixtral | 高效架构，GQA + Sliding Window Attention |
| Qwen 系列 | 阿里开源，中英文优化 |
| DeepSeek 系列 | 强推理能力，部分开源 |

## Encoder-Decoder：T5 / BART 系列

### 结构

Encoder-Decoder（也叫 Seq2Seq）保留了原始 Transformer 的完整结构：

- **Encoder**：双向注意力，编码输入序列为向量序列
- **Decoder**：因果注意力（自注意力部分），加上 Cross-Attention（关注 Encoder 的输出）
- **Cross-Attention**：Decoder 的每个位置都可以关注 Encoder 的所有位置，这是传递输入信息的桥梁

这种结构天然适合**输入和输出都是序列、但长度不必相等**的任务。

### T5：把一切表述为文本到文本

T5（Text-to-Text Transfer Transformer，2020）的核心思想是：把所有 NLP 任务统一表述为"输入文本 → 输出文本"。

```
翻译：  "translate English to French: That is good."  → "C'est bon."
摘要：  "summarize: <长文本>"                          → "<摘要>"
分类：  "sentiment: This movie is terrible."           → "negative"
问答：  "question: Who was Jim Henson? context: <段落>" → "a puppeteer"
```

T5 在 C4（Colossal Clean Crawled Corpus）上用这种统一格式预训练，然后在各任务上微调，在多个 benchmark 上达到当时 SOTA。

### BART：去噪自编码器

BART（2020）的预训练策略更有创意：对输入文本施加各种噪声（删除 token、打乱句子顺序、添加遮盖、随机旋转），让 Decoder 恢复原始文本。这让 BART 在生成任务（特别是摘要）上效果突出。

### 适合的任务

Encoder-Decoder 最适合**转换类任务**：

- **机器翻译**：输入源语言，输出目标语言
- **文本摘要**：输入长文，输出摘要（抽象式，非抽取式）
- **问答**：输入问题 + 上下文，输出答案（生成式）
- **文本纠错**：输入错误文本，输出正确文本
- **数据到文本**：输入结构化数据，输出描述文本

### 与 Decoder-Only 的竞争

随着 GPT-3 之后 Decoder-Only 大模型展示出强大的生成能力，Encoder-Decoder 的市场份额在逐渐收窄——因为一个足够大的 Decoder-Only 模型可以通过 prompt 完成翻译、摘要等任务，不再需要专门的 Encoder-Decoder 微调模型。

但在**资源受限场景**下（小模型、本地部署、任务明确），T5/BART 仍然是高效的选择。

## 选型指南

根据任务类型选择架构：

| 任务类型 | 典型任务 | 推荐架构 | 代表模型 |
|----------|----------|----------|----------|
| 理解/分类 | 情感分析、意图识别、主题分类 | Encoder-Only | BERT、RoBERTa、DeBERTa |
| 序列标注 | NER、词性标注、槽位填充 | Encoder-Only | BERT、MacBERT（中文） |
| 语义匹配 | 相似度计算、文本检索、去重 | Encoder-Only | BERT、Sentence-BERT |
| 开放域生成 | 对话、续写、创作 | Decoder-Only | GPT-2、LLaMA、Qwen |
| 指令跟随 | 问答、助手、Agent | Decoder-Only | GPT-4、LLaMA-3-Instruct |
| 机器翻译 | 多语言互译 | Encoder-Decoder | Helsinki-NLP/opus-mt、mBART |
| 文本摘要 | 长文压缩、新闻摘要 | Encoder-Decoder | BART、T5 |
| 生成式问答 | 基于段落生成答案 | Encoder-Decoder | T5、BART |
| Embedding 生成 | 向量检索、RAG | Encoder-Only | BGE、E5、text-embedding-ada |

**实用原则：**

- 任务标注数据充足 + 任务明确 → 选专门架构微调，更高效
- 任务多变 + 追求零样本/少样本能力 → 选 Decoder-Only 大模型
- 资源受限（本地 CPU 推理）→ 选 DistilBERT 或 T5-small
- 中文任务 → BERT 系选 MacBERT 或 chinese-roberta-wwm；生成选 Qwen

## 现代 LLM 的架构改进

原始 Transformer 的设计在扩展到百亿、千亿参数规模时，暴露了几个工程问题。现代大模型（LLaMA、Mistral 等）对原始设计做了几处关键改动。

### RoPE：旋转位置编码

**改了什么**：替代原始 Transformer 的绝对位置编码（正弦/余弦固定向量）。

**为什么改**：绝对位置编码的最大序列长度在预训练时固定，无法外推到更长的序列。RoPE（Rotary Position Embedding）把位置信息编码到 Query 和 Key 的旋转矩阵中，通过相对位置计算注意力，对序列长度的外推能力更好，且与注意力机制耦合更紧密。

LLaMA、Mistral、Qwen 等都使用 RoPE。

### RMSNorm：均方根归一化

**改了什么**：替代 LayerNorm（Layer Normalization）。

**为什么改**：LayerNorm 同时计算均值和方差进行归一化。RMSNorm 去掉均值中心化步骤，只用均方根（Root Mean Square）归一化，计算量更小，实验表明效果与 LayerNorm 相当甚至更好。LLaMA、Mistral、Qwen 均使用 Pre-RMSNorm（归一化放在残差连接之前）。

### SwiGLU：门控激活函数

**改了什么**：替代 FFN 层中的 ReLU 激活函数。

**为什么改**：SwiGLU = Swish + GLU（Gated Linear Unit），用门控机制控制信息流动，表达能力更强。实验（PaLM、LLaMA 论文）表明，在相同参数量下，SwiGLU 比 ReLU 和 GeLU 都有稳定的性能提升。代价是 FFN 层多一个线性变换，实际实现时通常把 FFN 隐层维度从 4x 调整到 ~2.67x 来保持总参数量不变。

### GQA：分组查询注意力

**改了什么**：替代标准多头注意力（MHA）中"每个头独立 KV"的设计。

**为什么改**：推理时 KV Cache 占用大量显存。标准 MHA 有 H 个 Query 头，也有 H 个 Key/Value 头。GQA（Grouped Query Attention）把 H 个 Query 头分成 G 组，每组共享一组 KV，KV 头数量从 H 降到 G（G << H）。这直接减少了 KV Cache 的大小，显著降低推理时的显存占用和 I/O 压力。LLaMA 3、Mistral 均使用 GQA。

MHA → MQA（极端情况，1 个 KV 头） → GQA（折中，G 个 KV 头）是一条工程演进路线，GQA 在质量和效率之间取得了较好的平衡。

---

这四处改动是当前主流开源大模型的标配。理解"改了什么"和"为什么改"，有助于读懂 LLaMA、Mistral 等模型的源码和论文。第 9 章讨论推理优化时，KV Cache 和 GQA 会再次出现，届时会从工程实现角度深入展开。
