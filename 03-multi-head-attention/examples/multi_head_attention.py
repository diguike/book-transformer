"""
Multi-Head Attention 实现
对应《Transformer 工程实战》第 3 章

参考论文：Attention Is All You Need (Vaswani et al., 2017)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 模块

    参数:
        d_model: 输入/输出的特征维度（原论文为 512）
        num_heads: 注意力头数（原论文为 8）

    要求: d_model 必须能被 num_heads 整除
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的维度
        self.d_k = d_model // num_heads

        # Q、K、V 的线性投影层
        # 注意：这里用一个大矩阵实现，等价于 num_heads 个小矩阵
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影层（W^O）
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一维切分成 (num_heads, d_k) 并调整维度顺序

        输入:  (batch, seq_len, d_model)
        输出:  (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        # reshape: (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # 交换 seq_len 和 num_heads，使得各头可以并行计算
        return x.transpose(1, 2)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Scaled Dot-Product Attention

        输入 q, k, v:  (batch, num_heads, seq_len, d_k)
        输出:          (batch, num_heads, seq_len, d_k)
        """
        # 计算注意力分数并缩放
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 可选的掩码（用于 Decoder 的 Causal Mask 或 Padding Mask）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, v)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向计算

        参数:
            q: Query，形状 (batch, seq_len_q, d_model)
            k: Key，  形状 (batch, seq_len_k, d_model)
            v: Value，形状 (batch, seq_len_k, d_model)
            mask: 可选掩码，形状 (batch, 1, seq_len_q, seq_len_k) 或可广播的形状

        返回:
            output: 形状 (batch, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # 第一步：线性投影
        q = self.w_q(q)  # (batch, seq_len_q, d_model)
        k = self.w_k(k)  # (batch, seq_len_k, d_model)
        v = self.w_v(v)  # (batch, seq_len_k, d_model)

        # 第二步：切分多头
        q = self.split_heads(q)  # (batch, num_heads, seq_len_q, d_k)
        k = self.split_heads(k)  # (batch, num_heads, seq_len_k, d_k)
        v = self.split_heads(v)  # (batch, num_heads, seq_len_k, d_k)

        # 第三步：各头并行计算 Attention
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        # attn_output: (batch, num_heads, seq_len_q, d_k)

        # 第四步：拼接各头结果
        # 先把维度调回 (batch, seq_len_q, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 再合并最后两维：(batch, seq_len_q, d_model)
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 第五步：输出投影
        output = self.w_o(attn_output)  # (batch, seq_len_q, d_model)

        return output


def test_output_shape():
    """验证输出 shape 正确"""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 自注意力：Q、K、V 来自同一序列
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x, x, x)

    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"期望输出 shape {expected_shape}，实际得到 {output.shape}"

    print(f"自注意力测试通过")
    print(f"  输入 shape:  {x.shape}")
    print(f"  输出 shape:  {output.shape}")

    # 交叉注意力：Q 来自解码器，K/V 来自编码器（序列长度可不同）
    seq_len_dec = 7
    seq_len_enc = 12
    q = torch.randn(batch_size, seq_len_dec, d_model)
    kv = torch.randn(batch_size, seq_len_enc, d_model)
    cross_output = mha(q, kv, kv)

    expected_cross_shape = (batch_size, seq_len_dec, d_model)
    assert cross_output.shape == expected_cross_shape, \
        f"期望交叉注意力输出 shape {expected_cross_shape}，实际得到 {cross_output.shape}"

    print(f"\n交叉注意力测试通过")
    print(f"  Q shape:     {q.shape}")
    print(f"  K/V shape:   {kv.shape}")
    print(f"  输出 shape:  {cross_output.shape}")


def test_with_mask():
    """验证带掩码时输出 shape 不变"""
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # 构造因果掩码（下三角矩阵），用于 Decoder
    # shape: (1, 1, seq_len, seq_len) 方便广播
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    output = mha(x, x, x, mask=causal_mask)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"带掩码的输出 shape 不符合预期，实际得到 {output.shape}"

    print(f"\n带因果掩码测试通过")
    print(f"  掩码 shape:  {causal_mask.shape}")
    print(f"  输出 shape:  {output.shape}")


def count_parameters():
    """统计参数量"""
    d_model = 512
    num_heads = 8
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    total = sum(p.numel() for p in mha.parameters())

    print(f"\n参数量统计 (d_model={d_model}, num_heads={num_heads})")
    print(f"  W_Q:    {mha.w_q.weight.numel():>8,}")
    print(f"  W_K:    {mha.w_k.weight.numel():>8,}")
    print(f"  W_V:    {mha.w_v.weight.numel():>8,}")
    print(f"  W_O:    {mha.w_o.weight.numel():>8,}")
    print(f"  总计:   {total:>8,}")
    print(f"  （等价于 {total // (d_model * d_model)} 个 {d_model}×{d_model} 矩阵）")


if __name__ == "__main__":
    print("=" * 50)
    print("Multi-Head Attention 测试")
    print("=" * 50)

    test_output_shape()
    test_with_mask()
    count_parameters()

    print("\n所有测试通过。")
