---
title: 附录 A 线性代数速查
feishu_url: "https://www.feishu.cn/wiki/QqyDwNGe0iDEmAkJI3JcioWwnof"
last_synced: "2026-05-07T23:07:51"
---

## 向量

### 基本定义

向量是一组有序数字，表示为列向量或行向量。在 Transformer 中，一个 token 的 embedding 就是一个向量，维度通常是 512 或 768。

$$\mathbf{v} = [v_1, v_2, \ldots, v_n]$$

NumPy 表示：

```python
import numpy as np

v = np.array([1.0, 2.0, 3.0])  # 形状 (3,)
```

### 向量运算

**加法**：逐元素相加，两个向量维度必须相同。

$$\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, \ldots, a_n + b_n]$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b  # array([5, 7, 9])
```

**数乘**：标量乘以向量，每个元素都乘以该标量。

$$\alpha \mathbf{v} = [\alpha v_1, \alpha v_2, \ldots, \alpha v_n]$$

```python
2 * a  # array([2, 4, 6])
```

**点积（内积）**：两个向量对应元素相乘再求和，结果是标量。

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

```python
np.dot(a, b)   # 32 = 1*4 + 2*5 + 3*6
# 或
a @ b          # 等价写法
```

**范数（向量长度）**

L2 范数（欧几里得范数）最常用：

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

```python
np.linalg.norm(v)        # L2 范数，默认
np.linalg.norm(v, ord=1) # L1 范数（绝对值之和）
```

**单位向量（归一化）**：

```python
v_normalized = v / np.linalg.norm(v)
# 归一化后 np.linalg.norm(v_normalized) ≈ 1.0
```

**余弦相似度**：

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 矩阵

### 形状（Shape）

矩阵 $\mathbf{A}$ 的形状记为 $(m, n)$，表示 $m$ 行 $n$ 列。在 Transformer 中，权重矩阵的形状标注是理解模型的基础。

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # 形状 (2, 3)

A.shape   # (2, 3)
A.ndim    # 2（维数）
A.size    # 6（总元素数）
```

### 转置

将矩阵的行列互换，$(m, n)$ 变为 $(n, m)$：

$$(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}$$

```python
A.T         # 形状 (3, 2)
np.transpose(A)  # 等价写法
```

### 矩阵乘法

矩阵 $\mathbf{A}$（形状 $m \times k$）乘以矩阵 $\mathbf{B}$（形状 $k \times n$），结果形状为 $m \times n$。

**维度规则**：$\mathbf{A}$ 的列数必须等于 $\mathbf{B}$ 的行数（中间维度对齐）。

$$(m \times k) \cdot (k \times n) = (m \times n)$$

```python
A = np.random.randn(3, 4)  # (3, 4)
B = np.random.randn(4, 5)  # (4, 5)
C = A @ B                  # (3, 5)
# 或
C = np.matmul(A, B)
```

**Transformer 中的典型矩阵乘法**：

```
Q: (seq_len, d_k)   × W_Q: (d_k, d_k) → Q: (seq_len, d_k)
K: (seq_len, d_k)   × W_K: (d_k, d_k) → K: (seq_len, d_k)
Q: (seq_len, d_k)   × K^T: (d_k, seq_len) → scores: (seq_len, seq_len)
```

### 逐元素乘法（Hadamard 积）

与矩阵乘法不同，逐元素乘法要求两个矩阵形状完全相同，对应位置相乘：

$$(\mathbf{A} \odot \mathbf{B})_{ij} = \mathbf{A}_{ij} \cdot \mathbf{B}_{ij}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A * B          # array([[ 5, 12], [21, 32]])
np.multiply(A, B)  # 等价写法
```

### 广播（Broadcasting）

NumPy 允许形状不完全相同的数组进行运算，自动扩展维度较小的数组：

```python
# 矩阵 + 向量（向量广播到每一行）
M = np.ones((3, 4))    # (3, 4)
v = np.array([1, 2, 3, 4])  # (4,)
M + v                  # (3, 4)，向量广播

# Transformer 中 softmax 除以 sqrt(d_k) 就是广播除法
scores = Q @ K.T       # (seq_len, seq_len)
scores / np.sqrt(d_k)  # 标量广播
```

---

## 常用操作的 NumPy 写法对照

| 操作 | 数学记号 | NumPy |
|------|---------|-------|
| 向量点积 | $\mathbf{a}^T \mathbf{b}$ | `np.dot(a, b)` 或 `a @ b` |
| 矩阵乘法 | $\mathbf{A}\mathbf{B}$ | `A @ B` 或 `np.matmul(A, B)` |
| 逐元素乘法 | $\mathbf{A} \odot \mathbf{B}$ | `A * B` |
| 矩阵转置 | $\mathbf{A}^T$ | `A.T` |
| L2 范数 | $\|\mathbf{v}\|_2$ | `np.linalg.norm(v)` |
| 矩阵形状 | $(m, n)$ | `A.shape` |
| 沿轴求和 | $\sum_j A_{ij}$ | `A.sum(axis=1)` |
| 沿轴求最大值 | $\max_j A_{ij}$ | `A.max(axis=1)` |
| 指数函数 | $e^{\mathbf{v}}$ | `np.exp(v)` |
| 拼接矩阵 | $[\mathbf{A}; \mathbf{B}]$（纵向） | `np.concatenate([A, B], axis=0)` |
| 拼接矩阵 | $[\mathbf{A}, \mathbf{B}]$（横向） | `np.concatenate([A, B], axis=1)` |
| 改变形状 | — | `A.reshape(m, n)` |
| 增加维度 | — | `A[np.newaxis, :]` 或 `np.expand_dims(A, 0)` |

---

## Softmax 函数

Softmax 将任意实数向量转换为概率分布（所有元素非负，且求和为 1）。Transformer 的 Attention 机制中，用 softmax 对 score 向量做归一化。

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**数值稳定的实现**（先减最大值，防止 `exp` 溢出）：

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    # 减去最大值确保数值稳定，不改变输出结果
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# 示例
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
# array([0.659, 0.242, 0.099])，求和为 1.0
```

**为什么减最大值不影响结果：**

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_j e^{x_j} \cdot e^{-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

PyTorch 中直接使用：

```python
import torch
import torch.nn.functional as F

x = torch.tensor([2.0, 1.0, 0.1])
F.softmax(x, dim=-1)
# tensor([0.6590, 0.2424, 0.0986])
```

---

## 读 Transformer 论文时的常见记号

阅读《Attention Is All You Need》及其后续论文时，以下记号高频出现：

| 记号 | 含义 | 典型值 |
|------|------|-------|
| $d_{\text{model}}$ | 模型的隐藏层维度，也叫 embedding 维度 | 512（Base），1024（Large） |
| $d_k$ | Query 和 Key 的向量维度 | $d_{\text{model}} / h$ |
| $d_v$ | Value 的向量维度 | $d_{\text{model}} / h$ |
| $h$ | Multi-Head Attention 的头数 | 8（Base），16（Large） |
| $n$ | 序列长度（token 数） | 随输入变化 |
| $N$ | Transformer 层数（编码器/解码器堆叠数） | 6（Base），12（Large） |
| $d_{ff}$ | Feed-Forward 层的中间维度 | $4 \times d_{\text{model}}$ |
| $V$ | 词表大小 | 约 30000-50000 |
| $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ | Q/K/V 的投影权重矩阵 | 形状均为 $(d_{\text{model}}, d_k)$ |
| $\mathbf{W}^O$ | 多头输出的投影矩阵 | 形状 $(h \cdot d_v, d_{\text{model}})$ |

**Scaled Dot-Product Attention 公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

各矩阵形状（batch 维度省略）：

```
Q: (n, d_k)
K: (n, d_k)
V: (n, d_v)

QK^T:         (n, n)
QK^T/sqrt(d_k): (n, n)
softmax(...):  (n, n)   ← 每行是对所有位置的注意力权重
Attention 输出: (n, d_v)
```

除以 $\sqrt{d_k}$ 是为了防止点积值过大导致 softmax 进入梯度趋近于零的饱和区。

---

> 本附录来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
