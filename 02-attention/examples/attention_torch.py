"""
第2章：用 PyTorch 实现 Scaled Dot-Product Attention
对比 numpy 版本，展示 PyTorch 写法的简洁性，并验证两者结果一致
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# numpy 版本（从 attention_scratch.py 复制，方便对比）
# ============================================================

def softmax_np(x: np.ndarray) -> np.ndarray:
    """numpy softmax，对最后一维操作"""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """numpy 实现"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax_np(scores)
    return weights @ V


# ============================================================
# PyTorch 版本
# ============================================================

def attention_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 实现的 Scaled Dot-Product Attention

    相比 numpy 版本，PyTorch 版本的优势：
    1. 支持 GPU 加速（只需把 tensor 移到 cuda 设备）
    2. 支持 mask（因果掩码、padding 掩码）
    3. 自动微分，可以直接用于训练
    4. 支持批量计算（batch 维度）

    参数:
        Q: [batch_size, seq_len, d_k] 或 [seq_len, d_k]
        K: [batch_size, seq_len, d_k] 或 [seq_len, d_k]
        V: [batch_size, seq_len, d_v] 或 [seq_len, d_v]
        mask: 可选，形状与 scores 兼容，0 的位置会被遮盖

    返回:
        output: 与 V 同形状
        weights: 注意力权重矩阵
    """
    d_k = Q.size(-1)

    # matmul 支持广播，自动处理 batch 维度
    # Q: [..., seq_len, d_k]  K.transpose(-2,-1): [..., d_k, seq_len]
    # -> scores: [..., seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 被 mask 掉的位置填充 -inf，经过 softmax 后权重趋近于 0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # F.softmax 在最后一维做 softmax，和 numpy 版本等价
    weights = F.softmax(scores, dim=-1)

    output = torch.matmul(weights, V)
    return output, weights


# ============================================================
# 对比两种实现的输出
# ============================================================

print("=" * 60)
print("对比 numpy 和 PyTorch 实现的输出")
print("=" * 60)

np.random.seed(42)
seq_len = 4
d_k = 8

# numpy 版输入
Q_np = np.random.randn(seq_len, d_k).astype(np.float32)
K_np = np.random.randn(seq_len, d_k).astype(np.float32)
V_np = np.random.randn(seq_len, d_k).astype(np.float32)

# 转换成 PyTorch tensor（共享同一份数据）
Q_pt = torch.from_numpy(Q_np)
K_pt = torch.from_numpy(K_np)
V_pt = torch.from_numpy(V_np)

# 分别计算
out_np = attention_numpy(Q_np, K_np, V_np)
out_pt, weights_pt = attention_torch(Q_pt, K_pt, V_pt)

print(f"numpy  输出 shape: {out_np.shape}")
print(f"PyTorch 输出 shape: {out_pt.shape}")

# 验证数值一致性（允许浮点误差）
max_diff = np.abs(out_np - out_pt.numpy()).max()
print(f"\n两种实现最大绝对误差: {max_diff:.2e}")
print(f"结果{'一致 ✓' if max_diff < 1e-5 else '不一致 ✗'}")

print()
print("=" * 60)
print("PyTorch 批量计算（batch 维度）")
print("=" * 60)

batch_size = 2

# 直接扩展到 batch 维度，numpy 需要手动循环，PyTorch 自动处理
Q_batch = torch.randn(batch_size, seq_len, d_k)
K_batch = torch.randn(batch_size, seq_len, d_k)
V_batch = torch.randn(batch_size, seq_len, d_k)

out_batch, weights_batch = attention_torch(Q_batch, K_batch, V_batch)
print(f"输入 Q shape: {Q_batch.shape}")
print(f"输出 shape: {out_batch.shape}")
print(f"注意力权重 shape: {weights_batch.shape}")

print()
print("=" * 60)
print("因果掩码（Causal Mask）—— Decoder 用到")
print("=" * 60)

# Decoder 在生成第 t 个 token 时，只能看到 0..t-1 位置的 token，不能看到未来
# 用上三角矩阵实现这个约束：1 表示可见，0 表示遮盖

# tril 保留下三角（包含对角线），即位置 i 只能看到 j <= i 的位置
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print("因果掩码（1=可见，0=遮盖）:")
print(causal_mask)

Q_dec = torch.randn(seq_len, d_k)
K_dec = torch.randn(seq_len, d_k)
V_dec = torch.randn(seq_len, d_k)

out_causal, weights_causal = attention_torch(Q_dec, K_dec, V_dec, mask=causal_mask)
print(f"\n应用因果掩码后的注意力权重（上三角应为 0）:")
print(weights_causal.round(decimals=4))
print("上三角（不含对角线）是否全为 0:", (weights_causal.triu(diagonal=1) < 1e-6).all().item())

print()
print("=" * 60)
print("PyTorch 内置实现")
print("=" * 60)

# PyTorch 2.0 起内置了优化版的 scaled_dot_product_attention
# 在支持 Flash Attention 的硬件上会自动使用更高效的实现
# is_causal=True 等价于手动传入因果掩码
out_builtin = F.scaled_dot_product_attention(Q_pt, K_pt, V_pt, is_causal=False)
max_diff_builtin = (out_pt - out_builtin).abs().max().item()
print(f"F.scaled_dot_product_attention 与手写版最大误差: {max_diff_builtin:.2e}")
print("实际生产代码中优先使用内置实现，而不是手写")
