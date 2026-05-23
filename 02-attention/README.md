---
title: Attention 机制
feishu_url: "https://www.feishu.cn/wiki/FfJBwg0HaiZeQqkgxyqctZuynud"
last_synced: "2026-05-08T00:23:57"
---

## Seq2Seq 的瓶颈

![Encoder 中的 tensor 传递](assets/encoder_with_tensors.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

在 Attention 出现之前，序列到序列（Seq2Seq）任务——比如机器翻译——用的是 Encoder-Decoder 结构。Encoder 是一个 RNN，逐步读入源语言的每个词，最终把整个句子的信息压缩进最后一个隐藏状态向量 `h_n`。Decoder 是另一个 RNN 从这个向量出发，逐词生成目标语言。

问题在那个压缩步骤上。不管源句子有多长，最终都要塞进一个固定维度的向量。对于短句子，这勉强够用；对于长句子，信息丢失严重——早期输入的词的信息在 RNN 的反复传递中被稀释，到最后几乎消失。

这不是 RNN 实现问题，是架构上的根本限制：单一向量的信息容量是有上限的。

2015 年 Bahdanau 等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中提出了 Attention 机制。核心思路是：不强迫 Encoder 把所有信息压进一个向量，而是保留 Encoder 每一步的隐藏状态，让 Decoder 在生成每个词时自己决定该"看"Encoder 的哪些位置。

## Attention 的核心思想

用翻译任务说明。把英文 "The cat sat on the mat" 翻译成法文，Decoder 在生成 "le chat"（猫）时，应该重点关注 Encoder 输出里对应 "the cat" 的位置；在生成 "s'est assis"（坐下）时，应该关注 "sat" 对应的位置。

Attention 就是给 Encoder 的每个输出位置计算一个权重，权重之和为 1，表示 Decoder 此刻"注意力"的分布。然后用这些权重对 Encoder 的输出做加权求和，得到一个上下文向量（context vector），再用这个向量指导当前词的生成。

每生成一个词，权重分布都重新计算一次，也就是说注意力是动态的，不是固定的。

Attention 的思路不止于此。Bahdanau 的方案只解决了 Decoder 看 Encoder 的问题。Transformer 把这个思路进一步泛化：序列内部的每个位置也可以对其他所有位置做 Attention，不再局限于跨序列的关注。

Transformer 里用的是自注意力（Self-Attention）：不是 Decoder 去关注 Encoder 输出，而是序列中每个位置对序列中所有其他位置计算 Attention，让每个位置的表示都能融入上下文信息。

## Q、K、V 是什么

Transformer 的 Attention 把输入拆成三个矩阵：**Query（Q）**、**Key（K）**、**Value（V）**。这三个名字来自信息检索的类比，用数据库查询来理解最直接。

假设有一个键值对数据库：

```
Key: "猫"   → Value: "关于猫的完整信息"
Key: "狗"   → Value: "关于狗的完整信息"
Key: "坐下" → Value: "关于坐下动作的信息"
```

你发出一个 Query："这里有动物吗？"

硬查询（精确匹配）只能匹配完全相同的 Key。但软查询（Attention）会计算 Query 和每个 Key 的相似度，得到每条记录的匹配分数，然后按分数加权混合所有 Value，返回一个融合了相关信息的结果。

在 Self-Attention 里：

- **Q**（Query）：当前位置想查什么
- **K**（Key）：每个位置"对外声明"自己是什么
- **V**（Value）：每个位置实际携带的内容

Q 和 K 用来计算相似度，决定注意力权重；V 是被加权求和的实际内容。Q 和 K 的维度必须相同（因为要做点积），V 的维度可以不同（通常和 K 相同）。

![Q、K、V 向量计算示意](assets/transformer_self_attention_vectors.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

Q、K、V 三个矩阵是从同一个输入 X 经过三个不同的线性变换得到的：

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

`W_Q`、`W_K`、`W_V` 是训练学到的参数矩阵。不同的投影让模型可以分别学习"如何提问"和"如何被查询"，而不是把这两个角色混在一起。

## Scaled Dot-Product Attention

完整公式：

```
Attention(Q, K, V) = softmax( Q · K^T / √d_k ) · V
```

每一步拆解：

**第一步：点积 `Q · K^T`**

计算每个 Query 和所有 Key 的相似度。结果是一个 `[seq_len, seq_len]` 的矩阵，每个元素 `(i, j)` 表示位置 `i` 的 Query 和位置 `j` 的 Key 的相似度得分。点积越大，说明两者越"匹配"。

![注意力得分计算](assets/transformer_self_attention_score.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

**第二步：除以 √d_k**

缩放步骤，单独一节讲。

**第三步：softmax**

把得分矩阵的每一行转成概率分布（每行和为 1）。这就是注意力权重：对于序列中的每个位置，它对其他所有位置的"关注程度"。

![Softmax 归一化注意力权重](assets/self-attention_softmax.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

**第四步：乘以 V**

用注意力权重对 V 做加权求和。结果是 `[seq_len, d_v]`，每个位置都得到了一个融合了全序列上下文的新表示。

![加权求和输出](assets/self-attention-output.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

用数字串联一遍：设 `seq_len=4, d_k=8`。

- `Q`: shape `[4, 8]`
- `K^T`: shape `[8, 4]`
- `Q · K^T`: shape `[4, 4]`——每对位置之间的相似度
- 除以 `√8 ≈ 2.83`，softmax 归一化
- 乘以 `V`（shape `[4, 8]`）：输出 shape `[4, 8]`

输入和输出的 shape 相同，但每个位置的向量内容已经混入了上下文信息。

## 为什么要除以 √d_k

点积的量级随维度增长。两个 `d_k` 维的随机向量（均值 0，方差 1，各分量独立同分布），它们点积的方差是 `d_k`，标准差是 `√d_k`。

当 `d_k` 很大时（比如 64、512），点积值很大。送进 softmax 后，最大值对应的位置权重趋近 1，其他位置趋近 0——softmax 进入饱和区。饱和区的梯度几乎为 0，反向传播时参数几乎不更新，训练困难。

除以 `√d_k` 把点积的方差缩回 1，softmax 的输入保持在合理范围，梯度流通顺畅。

这是一个简单但关键的工程细节，原论文《Attention Is All You Need》里专门做了实验验证：不除的版本在高维度时性能显著下降。

## 矩阵形式的 Attention

实际实现中，不会一次处理一个 Query，而是把整个序列的 Q、K、V 同时送进去，用矩阵乘法一次算出所有位置的 Attention，充分利用 GPU 的并行计算能力。

![矩阵形式 Attention 计算（步骤一）](assets/self-attention-matrix-calculation.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

![矩阵形式 Attention 计算（步骤二）](assets/self-attention-matrix-calculation-2.png)
*图片来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar*

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # 计算注意力得分，Q·K^T 除以 sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # mask shape: (seq_len, seq_len) 或 (batch, 1, seq_len, seq_len)
        # mask 为 0 的位置填充极大负值，softmax 后权重趋近于 0
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # softmax 得到注意力权重
    weights = F.softmax(scores, dim=-1)
    # 加权求和 V
    output = torch.matmul(weights, V)
    return output, weights
```

这就是 Transformer 里 Attention 最核心的计算，完整实现只需要几行。

---

## TypeScript 对照

对 JavaScript/TypeScript 背景的工程师，下面是同一个函数的 TypeScript 实现。纯数值计算，不依赖任何库，用二维数组模拟矩阵操作。这个实现仅用于理解原理，不适合生产使用。

```typescript
/**
 * 矩阵转置
 * [[1,2],[3,4]] -> [[1,3],[2,4]]
 */
function transpose(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix[0].length;
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => matrix[i][j])
  );
}

/**
 * 矩阵乘法
 * a: [m, k], b: [k, n] -> result: [m, n]
 */
function matMul(a: number[][], b: number[][]): number[][] {
  const m = a.length;
  const k = a[0].length;
  const n = b[0].length;
  const result = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let p = 0; p < k; p++) {
        result[i][j] += a[i][p] * b[p][j];
      }
    }
  }
  return result;
}

/**
 * 对二维矩阵的每一行做 softmax
 */
function softmax(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const maxVal = Math.max(...row); // 数值稳定性：减去最大值再 exp
    const exps = row.map((x) => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  });
}

/**
 * Scaled Dot-Product Attention
 * Q: [seq_len, d_k]
 * K: [seq_len, d_k]
 * V: [seq_len, d_v]
 * 返回: [seq_len, d_v]
 */
function scaledDotProductAttention(
  Q: number[][],
  K: number[][],
  V: number[][]
): number[][] {
  const d_k = Q[0].length;

  // 第一步：Q · K^T，得到 [seq_len, seq_len] 的得分矩阵
  const scores = matMul(Q, transpose(K));

  // 第二步：除以 sqrt(d_k) 防止点积过大导致 softmax 饱和
  const scale = Math.sqrt(d_k);
  const scaledScores = scores.map((row) => row.map((x) => x / scale));

  // 第三步：softmax，每行变成概率分布
  const weights = softmax(scaledScores);

  // 第四步：加权求和 V，输出 [seq_len, d_v]
  return matMul(weights, V);
}

// 简单验证：seq_len=3, d_k=4
const Q = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
const K = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
const V = [[1, 2], [3, 4], [5, 6]];

const output = scaledDotProductAttention(Q, K, V);
console.log("output shape:", output.length, "x", output[0].length); // 3 x 2
console.log("output:", output);
```

TS 版本和 Python/numpy 版本逻辑完全一致。区别只在于：Python 用 `@` 运算符或 `numpy.matmul` 完成矩阵乘法，底层由 BLAS 库执行，速度快几个数量级；TS 版本是手写三重循环，适合对照理解计算过程。

---

代码示例在 `examples/` 目录下，需要先安装依赖：

```bash
pip install -r examples/requirements.txt
```

- `attention_scratch.py`：用 numpy 从零实现 Scaled Dot-Product Attention
- `attention_torch.py`：用 PyTorch 实现，并与 numpy 版本结果对比
