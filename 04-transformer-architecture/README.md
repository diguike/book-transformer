---
title: Transformer 完整架构
feishu_url: "https://www.feishu.cn/wiki/RUNXwZCKviy0J9kR7TJctbzLnaf"
last_synced: "2026-05-08T00:24:18"
---

## 整体结构回顾

原始 Transformer 论文（Vaswani et al., 2017）提出的架构由两个部分组成：Encoder 和 Decoder，各由 N 个相同结构的 Block 堆叠而成。原始论文取 `N=6`。

![Transformer 整体架构](assets/the_transformer_3.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

```
输入序列                          目标序列（训练时）
    ↓                                   ↓
[Input Embedding]               [Output Embedding]
    ↓                                   ↓
[Positional Encoding]           [Positional Encoding]
    ↓                                   ↓
┌─────────────────┐             ┌─────────────────────────────┐
│  Encoder Block  │ × N         │      Decoder Block          │ × N
│                 │             │                             │
│  Multi-Head     │             │  Masked Multi-Head Attn     │
│  Attention      │             │        ↓                    │
│      ↓          │      ───→   │  Cross-Attention            │
│  Add & Norm     │  (K, V 传递) │  (K, V 来自 Encoder)        │
│      ↓          │             │        ↓                    │
│  Feed-Forward   │             │  Feed-Forward               │
│      ↓          │             │        ↓                    │
│  Add & Norm     │             │  Add & Norm（每层）          │
└─────────────────┘             └─────────────────────────────┘
    ↓                                   ↓
（Encoder 输出）                 [Linear + Softmax]
                                        ↓
                                   输出概率分布
```

![Encoder-Decoder 整体示意](assets/The_transformer_encoders_decoders.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![Encoder 和 Decoder 堆叠结构](assets/The_transformer_encoder_decoder_stack.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

Encoder 负责理解输入序列，把每个位置编码成包含上下文信息的向量。Decoder 负责逐步生成目标序列，每一步借助 Encoder 的输出和已生成的部分序列来预测下一个 token。

整个架构是端对端可微的，可以直接用梯度下降训练。

## 位置编码

Attention 机制本身是置换不变的（permutation-invariant）：把输入序列的 token 打乱顺序，Attention 的计算结果只是相应调整，但模型无法区分位置 1 和位置 5 的 token 有什么本质差别。这导致 Transformer 天然感知不到序列的顺序信息。

解决方法是在送入 Encoder/Decoder 之前，把每个 token 的 embedding 加上一个位置相关的向量——**位置编码（Positional Encoding）**。

原始论文使用正弦/余弦函数构造位置编码：

![位置编码向量示意](assets/transformer_positional_encoding_vectors.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

```
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

![位置编码示例（小尺度）](assets/transformer_positional_encoding_example.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![位置编码示例（大尺度热力图）](assets/transformer_positional_encoding_large_example.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

其中 `pos` 是序列中的位置（0, 1, 2, ...），`i` 是维度索引（0, 1, ..., d_model/2 - 1）。

这个公式的直觉：不同维度使用不同频率的正弦/余弦波。低维度（小 `i`）频率高，能区分相邻位置；高维度（大 `i`）频率低，能捕捉长距离的位置关系。每个位置得到一个唯一的 `d_model` 维向量。

**为什么不用可学习的位置编码？**

论文同时实验了可学习的位置编码（每个位置一组可训练参数），发现两种方式效果相近。原始论文最终选择正弦编码，理由是：正弦编码是确定性的，不依赖训练数据，对比可学习编码，它可以外推到训练时未见过的更长序列。在早期资源有限的实验条件下，这是一个稳健的默认选择。

很多后续模型（将在第 5 章介绍）改用可学习的位置编码，因为长序列外推不再是主要约束，更重要的是让模型从数据中自行学习最优的位置表示。两种方案各有权衡，没有绝对的优劣之分。

## Encoder Block 内部

每个 Encoder Block 包含两个子层，每个子层后面跟一个残差连接加 Layer Norm：

```
输入 x
  ↓
x = LayerNorm(x + MultiHeadAttention(x, x, x))
  ↓
x = LayerNorm(x + FeedForward(x))
  ↓
输出 x
```

![Encoder Block 内部结构](assets/Transformer_encoder.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

第一个子层是 Multi-Head Self-Attention，Q、K、V 都来自同一个输入序列，所以叫"自注意力"。每个位置都能看到序列中所有其他位置的信息。

第二个子层是 Feed-Forward 网络（FFN），对每个位置独立做相同的两层 MLP 变换。

两个子层的输入输出维度都保持 `d_model`，这使得 Block 可以任意堆叠。

## 残差连接

每个子层的输出不是直接使用，而是加上输入之后再归一化：

```
output = LayerNorm(x + SubLayer(x))
```

![残差连接与 Layer Norm（单层）](assets/transformer_resideual_layer_norm.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![残差连接与 Layer Norm（Encoder 视角）](assets/transformer_resideual_layer_norm_2.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![残差连接与 Layer Norm（完整 Encoder-Decoder）](assets/transformer_resideual_layer_norm_3.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

这就是残差连接（Residual Connection），来自 ResNet（He et al., 2016）。

作用在于训练深层网络时的梯度稳定性。假设没有残差连接，梯度从输出层反向传播到输入层需要连续经过 N 个 Block 的所有非线性变换，在网络较深时容易出现梯度消失。

有了残差连接，梯度有一条"短路"路径可以直接通过加法操作传回去，乘以的系数始终是 1，不会被连续衰减。这让深层 Transformer（N=6 甚至更深）的训练变得稳定。

ResNet 是 He et al. 在 CVPR 2016 发表的深度卷积网络，通过残差连接解决了网络越深、训练越难的问题——这个思路被 Transformer 直接借用了。类比 ResNet：ResNet 学习的是残差 `F(x) = H(x) - x`，而不是直接学习目标映射 `H(x)`。如果某一层实际上不需要做太多变换，残差路径允许它趋近于恒等映射，权重向零收缩即可。

## Layer Normalization

归一化有多种方式，Transformer 选择 Layer Normalization（Ba et al., 2016），而不是 Batch Normalization（BN）。

**BN 的问题**：BN 在 batch 维度做归一化，计算均值和方差需要整个 batch 的数据。对于 NLP 任务，序列长度不固定，且 batch 内不同样本的语义差异很大，用 batch 统计量归一化会引入噪声。更重要的是，BN 在推理时依赖训练集的滑动统计量，在序列长度变化或小 batch 场景下表现不稳定。

**LayerNorm 的做法**：对每个样本、每个位置，在 `d_model` 维度上独立计算均值和方差，然后归一化：

```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

其中 `μ` 和 `σ` 是当前样本当前位置在 `d_model` 维度上的均值和标准差，`γ`、`β` 是可学习的缩放和偏移参数，`ε` 是防止除零的小常数。

LayerNorm 的计算只依赖当前位置的特征，不需要 batch 内其他样本的信息，因此在变长序列、在线推理等场景下表现稳定。

## Feed-Forward 层

Encoder Block 和 Decoder Block 里都有 Feed-Forward 网络（FFN），结构是两层线性变换加一个激活函数：

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

（原始论文用的是 ReLU，即上式中的 `max(0, ...)`；现代模型如 BERT、GPT-2 之后普遍改用 GELU，它在 0 附近更平滑，实践中效果略好。用 HuggingFace 加载这些模型时会看到 GELU。）

维度变化：`d_model → d_ff → d_model`。原始论文 `d_model=512`，`d_ff=2048`，中间层维度是输入的 4 倍。

**关键特性**：FFN 对序列中的**每个位置独立**地施加相同的变换。不同位置共享同一组参数（`W_1`, `W_2`, `b_1`, `b_2`），但各自的计算相互独立，不涉及位置间的信息交换。

信息交换只发生在 Attention 层，FFN 承担的是在每个位置对特征做非线性变换和升维处理。这种分工让两个子层各司其职：Attention 聚合上下文，FFN 在每个位置做特征变换。

FFN 的参数量占 Transformer 总参数量的很大比例。以 `d_model=512, d_ff=2048` 为例，单个 FFN 的参数约 `2 × 512 × 2048 = 2,097,152`，而单个 Multi-Head Attention 约 `4 × 512 × 512 = 1,048,576`，FFN 是 Attention 的两倍。

## Decoder Block 内部

Decoder Block 比 Encoder Block 多一个子层，共三个子层：

```
输入 x（来自上一个 Decoder Block 或目标序列 embedding）
  ↓
x = LayerNorm(x + MaskedMultiHeadAttention(x, x, x))   # 第一子层：Masked Self-Attention
  ↓
x = LayerNorm(x + CrossAttention(x, enc_output, enc_output))  # 第二子层：Cross-Attention
  ↓
x = LayerNorm(x + FeedForward(x))                       # 第三子层：FFN
  ↓
输出 x
```

![Decoder Block 内部结构](assets/Transformer_decoder.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

**第一子层：Masked Self-Attention**

和 Encoder 的 Self-Attention 类似，但加了因果掩码（Causal Mask），确保生成第 `t` 个 token 时，只能看到位置 0 到 `t-1` 的信息，不能"偷看"未来的 token。

**第二子层：Cross-Attention（交叉注意力）**

Query 来自 Decoder 当前层的输出，Key 和 Value 来自 Encoder 的最终输出。这是 Decoder"查询"Encoder 输出的机制，让 Decoder 在生成每个 token 时能参考完整的输入序列信息。

**第三子层：FFN**

与 Encoder Block 的 FFN 相同，对每个位置独立做特征变换。

## Causal Mask

生成任务要求自回归（Autoregressive）：第 `t` 步只能基于前 `t-1` 步的输出做预测。如果在 Self-Attention 里让 Decoder 看到未来的 token，就相当于训练时"作弊"——模型学到了直接复制答案，而不是真正学会生成。

Causal Mask 是一个下三角矩阵，形状 `(seq_len, seq_len)`：

```
位置 0: [1, 0, 0, 0, 0]
位置 1: [1, 1, 0, 0, 0]
位置 2: [1, 1, 1, 0, 0]
位置 3: [1, 1, 1, 1, 0]
位置 4: [1, 1, 1, 1, 1]
```

值为 0 的位置在 Attention 计算时被填充为 `-inf`，经过 softmax 之后注意力权重变为 0，效果等同于"看不见"这些位置。

（注：第 5 章在描述 GPT 系列架构时，用"上三角区域填充为 -∞"来表述同一机制——两种说法等价：下三角为 1 的位置保留计算，对应上三角为 0 的位置填充 -∞ 后权重归零。）

PyTorch 实现：

```python
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

## 完整数据流：一个 Token 的旅程

以翻译任务为例，追踪一个 token 从输入到输出的完整路径。

**输入侧（Encoder）：**

1. 输入序列 `["The", "cat", "sat"]` 经过词表映射，得到 token id 序列，例如 `[234, 891, 1204]`
2. 查 Embedding 表，每个 id 映射为 `d_model` 维的向量
3. 加上位置编码，每个位置得到一个独特的向量
4. 进入第 1 个 Encoder Block：Self-Attention（看到所有位置）→ Add & Norm → FFN → Add & Norm
5. 输出传入第 2 个 Encoder Block，以此类推，共 N 次
6. 第 N 个 Encoder Block 的输出作为 Encoder 的最终表示，维度 `(seq_len, d_model)`

![自回归解码过程（动图第一帧）](assets/transformer_decoding_1.gif)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![自回归解码过程（动图第二帧）](assets/transformer_decoding_2.gif)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

**输出侧（Decoder）：**

7. 解码从 `<BOS>`（Begin of Sequence）token 开始
8. `<BOS>` 经过 Embedding + 位置编码，进入 Decoder Block
9. 第一子层：Masked Self-Attention（目前只有 `<BOS>` 一个 token，掩码暂时不起作用）
10. 第二子层：Cross-Attention，Decoder 以当前解码状态为 Query，动态查询 Encoder 输出中的相关信息，而不是一次性把 Encoder 输出全部读入
11. 第三子层：FFN 做特征变换
12. 经过 N 个 Decoder Block 后，输出进入线性层（维度 `d_model → vocab_size`）加 Softmax，得到词表上的概率分布
13. 取概率最大的 token（贪心解码）或按概率采样，得到第一个输出 token

14. 将已生成的 token 拼接到 Decoder 输入，重复步骤 8-13，直到生成 `<EOS>` 或达到最大长度

这个过程中，Encoder 只需要跑一次，Cross-Attention 每步都会"回看"同样的 Encoder 输出。

---

代码示例在 `examples/` 目录下，需要先安装依赖：

```bash
pip install -r examples/requirements.txt
```

- `positional_encoding.py`：实现 `PositionalEncoding` 类，打印前 20 个位置的编码值，验证余弦相似度随距离增大而减小
- `encoder_block.py`：实现 `EncoderBlock` 和 `Encoder`，6 层堆叠，验证输入输出 shape
- `transformer_full.py`：完整 Transformer 实现，包含 Decoder，演示自回归生成过程

---

> 本章来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
