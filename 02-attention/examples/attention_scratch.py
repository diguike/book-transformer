"""
第2章：用 numpy 从零实现 Scaled Dot-Product Attention
不依赖深度学习框架，直接用矩阵运算展示每一步计算过程
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    对矩阵的最后一个维度做 softmax
    减去最大值是为了数值稳定，防止 exp 溢出
    """
    # x.max(axis=-1, keepdims=True) 对每行取最大值，保持维度用于广播
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention 的 numpy 实现

    参数:
        Q: Query 矩阵，shape [seq_len, d_k]
        K: Key 矩阵，shape [seq_len, d_k]
        V: Value 矩阵，shape [seq_len, d_v]

    返回:
        output: 加权求和的输出，shape [seq_len, d_v]
        weights: 注意力权重，shape [seq_len, seq_len]
    """
    d_k = Q.shape[-1]  # Key 的维度，用于缩放

    # 第一步：计算 Q 和 K^T 的点积，得到相似度得分矩阵
    # Q: [seq_len, d_k]  K.T: [d_k, seq_len]  -> scores: [seq_len, seq_len]
    # scores[i, j] 表示位置 i 的 Query 与位置 j 的 Key 的相似度
    scores = Q @ K.T  # 等价于 np.matmul(Q, K.T)

    # 第二步：除以 sqrt(d_k) 防止点积值过大导致 softmax 饱和
    # 随机向量点积的标准差约为 sqrt(d_k)，除以它可以把方差归一化到 1
    scores = scores / np.sqrt(d_k)

    # 第三步：softmax 归一化，每行变成和为 1 的概率分布
    # weights[i, :] 是位置 i 对所有位置的注意力权重
    weights = softmax(scores)

    # 第四步：用注意力权重对 V 做加权求和
    # weights: [seq_len, seq_len]  V: [seq_len, d_v]  -> output: [seq_len, d_v]
    # output[i] = sum_j(weights[i, j] * V[j, :])
    output = weights @ V

    return output, weights


# ============================================================
# 构造一个简单例子并运行
# ============================================================

# 设置随机种子，保证结果可复现
np.random.seed(42)

seq_len = 4   # 序列长度（token 数量）
d_k = 8       # Q 和 K 的维度
d_v = 8       # V 的维度（通常和 d_k 相同）

# 随机生成 Q、K、V 矩阵（实际训练中这些是通过线性变换得到的）
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

print("=" * 60)
print("输入")
print("=" * 60)
print(f"Q shape: {Q.shape}  (seq_len={seq_len}, d_k={d_k})")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

print()
print("=" * 60)
print("逐步计算过程")
print("=" * 60)

# 步骤展开，便于观察中间结果
scores_raw = Q @ K.T
print(f"Q · K^T（原始得分）shape: {scores_raw.shape}")
print("原始得分矩阵（行=Query位置，列=Key位置）:")
print(np.round(scores_raw, 3))

scores_scaled = scores_raw / np.sqrt(d_k)
print(f"\n除以 sqrt({d_k})={np.sqrt(d_k):.3f} 后:")
print(np.round(scores_scaled, 3))

weights = softmax(scores_scaled)
print(f"\nsoftmax 后的注意力权重（每行和为1）:")
print(np.round(weights, 3))
print(f"每行之和: {weights.sum(axis=-1).round(6)}")  # 应为全1

output, _ = scaled_dot_product_attention(Q, K, V)
print(f"\n最终输出 shape: {output.shape}")
print("前两行:")
print(np.round(output[:2], 4))

print()
print("=" * 60)
print("语义验证：相同 token 的注意力得分应最高")
print("=" * 60)

# 构造一个可解释的例子：让 Q 和 K 有明确的语义
# 直接用 np.eye(seq_len, d_k) 构造正交向量
# 每个位置得到一个标准基向量（在 d_k 维空间中）
# 不同位置的向量彼此正交，因此与自身的点积最大，与其他位置的点积为 0
Q_eye = np.eye(seq_len, d_k)   # shape [seq_len, d_k]，每行是一个标准基向量
K_eye = np.eye(seq_len, d_k)

# 相同位置的 Q 和 K 完全一致（且互相正交），对角线得分应最大
scores_eye = Q_eye @ K_eye.T / np.sqrt(d_k)
weights_eye = softmax(scores_eye)
print("当每个位置的 Q 和 K 彼此正交时，对角线权重应最大：")
print(np.round(weights_eye, 3))
print("（对角线是每个位置对自身的注意力权重）")
