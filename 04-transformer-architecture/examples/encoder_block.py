"""
Encoder Block 和 Encoder 实现
对应《Transformer 工程实战》第 4 章

结构：
    EncoderBlock: Multi-Head Attention → Add & Norm → FFN → Add & Norm
    Encoder:      N 个 EncoderBlock 堆叠
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用第 3 章的 MultiHeadAttention
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../03-multi-head-attention/examples'))
from multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    每个位置独立的两层 MLP：d_model → d_ff → d_model
    原始论文 d_model=512, d_ff=2048（4 倍关系）
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # ReLU 激活，原始论文用 ReLU，后续工作常用 GELU
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    """
    单个 Encoder Block

    子层：
        1. Multi-Head Self-Attention + Add & Norm
        2. Feed-Forward + Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # 两个 Layer Norm，分别在 Attention 和 FFN 之后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x:    (batch, seq_len, d_model)
            mask: 可选的 padding mask

        返回:
            (batch, seq_len, d_model)
        """
        # 子层 1：Self-Attention + 残差 + LayerNorm
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层 2：FFN + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class Encoder(nn.Module):
    """
    完整 Encoder：N 个 EncoderBlock 堆叠

    参数:
        d_model:    特征维度（原始论文 512）
        num_heads:  注意力头数（原始论文 8）
        d_ff:       FFN 中间层维度（原始论文 2048）
        num_layers: Block 数量（原始论文 N=6）
        dropout:    dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        # 最后一个 LayerNorm（Post-LN 架构中 Encoder 输出后加）
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x:    (batch, seq_len, d_model)，已加完位置编码
            mask: 可选 padding mask

        返回:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


def test_encoder_block():
    """验证 EncoderBlock 输出 shape"""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    block = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    expected = (batch_size, seq_len, d_model)
    assert output.shape == expected, \
        f"EncoderBlock 输出 shape 不符合预期，期望 {expected}，实际 {output.shape}"

    print(f"EncoderBlock 测试通过")
    print(f"  输入 shape:  {x.shape}")
    print(f"  输出 shape:  {output.shape}")


def test_encoder():
    """验证 Encoder 输出 shape"""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    encoder = Encoder(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)

    expected = (batch_size, seq_len, d_model)
    assert output.shape == expected, \
        f"Encoder 输出 shape 不符合预期，期望 {expected}，实际 {output.shape}"

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder 测试通过（{num_layers} 层）")
    print(f"  输入 shape:  {x.shape}")
    print(f"  输出 shape:  {output.shape}")
    print(f"  总参数量:    {total_params:,}")


if __name__ == "__main__":
    print("=" * 50)
    print("Encoder Block 和 Encoder 测试")
    print("=" * 50)

    test_encoder_block()
    test_encoder()

    print("\n所有测试通过。")
