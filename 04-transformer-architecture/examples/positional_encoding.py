"""
位置编码（Positional Encoding）实现
对应《Transformer 工程实战》第 4 章

公式（原始论文）：
    PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
"""

import math
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码

    参数:
        d_model:   特征维度
        max_len:   支持的最大序列长度（预先计算好，推理时直接截取）
        dropout:   dropout 概率（正则化用，默认关闭）
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算 max_len 个位置的编码
        # pe 形状: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        # 对应公式中的 1 / 10000^(2i/d_model)，取对数再 exp 保证数值稳定
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度: (1, max_len, d_model)，方便广播加法
        pe = pe.unsqueeze(0)

        # 注册为 buffer：不是模型参数（不参与梯度更新），但会随模型保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码加到输入上

        参数:
            x: 形状 (batch, seq_len, d_model)

        返回:
            形状 (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def print_encoding_table():
    """打印前 20 个位置、前 8 个维度的编码值"""
    d_model = 64
    max_positions = 20
    show_dims = 8

    pe_module = PositionalEncoding(d_model=d_model)
    # pe 形状: (1, max_len, d_model)
    pe_values = pe_module.pe[0].numpy()  # (max_len, d_model)

    print("位置编码矩阵（前 20 个位置 × 前 8 个维度）")
    print(f"{'pos':>5}", end="")
    for d in range(show_dims):
        print(f"  dim{d:02d}", end="")
    print()
    print("-" * (5 + show_dims * 7))

    for pos in range(max_positions):
        print(f"{pos:>5}", end="")
        for d in range(show_dims):
            print(f"  {pe_values[pos, d]:+.3f}", end="")
        print()


def verify_cosine_similarity():
    """
    验证：相邻位置的余弦相似度应随距离增大而减小

    直觉：位置编码用不同频率的正弦波编码位置，
    相邻位置的编码向量更相似，距离越远越不同。
    """
    d_model = 512
    pe_module = PositionalEncoding(d_model=d_model)
    pe_values = pe_module.pe[0].numpy()  # (max_len, d_model)

    # 以位置 0 为基准，计算它与各距离位置的余弦相似度
    base_pos = 0
    anchor = pe_values[base_pos]
    anchor_norm = np.linalg.norm(anchor)

    print(f"\n位置 {base_pos} 与其他位置的余弦相似度：")
    distances = [1, 2, 4, 8, 16, 32, 64, 128]
    similarities = []

    for dist in distances:
        target = pe_values[dist]
        target_norm = np.linalg.norm(target)
        cosine_sim = np.dot(anchor, target) / (anchor_norm * target_norm)
        similarities.append(cosine_sim)
        print(f"  距离 {dist:>4}: 余弦相似度 = {cosine_sim:.4f}")

    # 验证相似度随距离单调递减
    is_decreasing = all(
        similarities[i] >= similarities[i + 1]
        for i in range(len(similarities) - 1)
    )
    print(f"\n相似度随距离增大而减小: {is_decreasing}")
    assert is_decreasing, "位置编码的相似度应随距离增大而减小"


if __name__ == "__main__":
    print("=" * 60)
    print("位置编码验证")
    print("=" * 60)

    print_encoding_table()
    verify_cosine_similarity()

    # 验证 forward 的 shape
    batch_size, seq_len, d_model = 2, 30, 64
    pe = PositionalEncoding(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe(x)
    assert out.shape == (batch_size, seq_len, d_model), \
        f"输出 shape 不符合预期: {out.shape}"
    print(f"\nforward shape 测试通过: {x.shape} → {out.shape}")
