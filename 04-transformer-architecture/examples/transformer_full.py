"""
简化版完整 Transformer（Encoder + Decoder）
对应《Transformer 工程实战》第 4 章

目标：展示完整数据流，使用 toy 翻译任务演示自回归生成过程。
注意：这是教学代码，去掉了部分工程细节（如 label smoothing、
      beam search），重点在于结构清晰。
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '../../03-multi-head-attention/examples'))
from multi_head_attention import MultiHeadAttention
from encoder_block import FeedForward, Encoder
from positional_encoding import PositionalEncoding


# ─────────────────────────────────────────
# Decoder Block
# ─────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    单个 Decoder Block，包含三个子层：
        1. Masked Self-Attention（因果掩码，只看历史）
        2. Cross-Attention（Query 来自 Decoder，K/V 来自 Encoder 输出）
        3. Feed-Forward

    每个子层后面跟 Add & Norm。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 子层 1：Masked Self-Attention（只看已生成的 token）
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # 子层 2：Cross-Attention（与 Encoder 输出交互）
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # 子层 3：FFN
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数:
            x:          (batch, tgt_len, d_model)，当前 Decoder 输入
            enc_output: (batch, src_len, d_model)，Encoder 的最终输出
            src_mask:   Encoder 侧的 padding mask（可选）
            tgt_mask:   因果掩码，形状 (1, 1, tgt_len, tgt_len)

        返回:
            (batch, tgt_len, d_model)
        """
        # 子层 1：Masked Self-Attention
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 子层 2：Cross-Attention，Q 来自 Decoder，K/V 来自 Encoder
        cross_attn_out = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 子层 3：FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class Decoder(nn.Module):
    """N 个 DecoderBlock 堆叠"""

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
            DecoderBlock(d_model=d_model, num_heads=num_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)


# ─────────────────────────────────────────
# 完整 Transformer
# ─────────────────────────────────────────

class Transformer(nn.Module):
    """
    简化版完整 Transformer

    参数:
        src_vocab_size: 源语言词表大小
        tgt_vocab_size: 目标语言词表大小
        d_model:        特征维度（原始论文 512）
        num_heads:      注意力头数（原始论文 8）
        d_ff:           FFN 中间层维度（原始论文 2048）
        num_layers:     Encoder/Decoder 的层数（原始论文 6）
        max_len:        支持的最大序列长度
        dropout:        dropout 概率
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Embedding 层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码（Encoder 和 Decoder 共用一个 PE 模块，参数相同）
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len,
                                               dropout=dropout)

        # Encoder 和 Decoder
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads,
                               d_ff=d_ff, num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads,
                               d_ff=d_ff, num_layers=num_layers, dropout=dropout)

        # 输出投影：d_model → tgt_vocab_size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 原始论文中 Embedding 权重与输出投影共享，这里简化不做共享
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化，原始论文的做法"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构造因果掩码（下三角矩阵）

        返回:
            (1, 1, seq_len, seq_len)，可广播到 (batch, num_heads, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        编码源序列

        参数:
            src:      (batch, src_len)，token id 序列
            src_mask: 可选 padding mask

        返回:
            (batch, src_len, d_model)
        """
        # Embedding + 位置编码
        # 原始论文中 Embedding 权重乘以 sqrt(d_model)，稳定训练初期的梯度
        x = self.src_embedding(src) * (self.src_embedding.embedding_dim ** 0.5)
        x = self.pos_encoding(x)
        return self.encoder(x, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        解码目标序列（训练时一次性解码整个序列，推理时逐步调用）

        参数:
            tgt:        (batch, tgt_len)，目标 token id
            enc_output: (batch, src_len, d_model)，Encoder 输出
            src_mask:   可选 padding mask

        返回:
            logits (batch, tgt_len, tgt_vocab_size)
        """
        tgt_len = tgt.size(1)
        tgt_mask = self.make_causal_mask(tgt_len, device=tgt.device)

        x = self.tgt_embedding(tgt) * (self.tgt_embedding.embedding_dim ** 0.5)
        x = self.pos_encoding(x)
        dec_output = self.decoder(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.output_projection(dec_output)
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        训练时的前向计算

        参数:
            src: (batch, src_len)
            tgt: (batch, tgt_len)

        返回:
            logits (batch, tgt_len, tgt_vocab_size)
        """
        enc_output = self.encode(src, src_mask=src_mask)
        logits = self.decode(tgt, enc_output, src_mask=src_mask)
        return logits


# ─────────────────────────────────────────
# Toy 翻译任务演示
# ─────────────────────────────────────────

def demo_autoregressive_generation():
    """
    用 toy 任务演示自回归生成过程

    任务设定：输入序列 [1,2,3,4]，Decoder 逐步生成输出。
    词表很小，重点展示数据流，不追求有意义的输出。
    """
    # 超参数（缩小便于快速运行）
    src_vocab_size = 10
    tgt_vocab_size = 10
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    BOS_ID = 1   # Begin of Sequence token
    EOS_ID = 2   # End of Sequence token
    MAX_GEN_LEN = 8

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=0.0,  # 推理时关掉 dropout
    )
    model.eval()

    # 输入序列（未训练，权重随机，输出无意义，但数据流是正确的）
    src = torch.tensor([[1, 2, 3, 4]])  # (batch=1, src_len=4)

    print("Toy 翻译任务演示（随机初始化，展示数据流）")
    print(f"  输入序列: {src[0].tolist()}")
    print()

    with torch.no_grad():
        # 第一步：Encoder 只跑一次，得到 enc_output
        enc_output = model.encode(src)
        print(f"Encoder 输出 shape: {enc_output.shape}")
        # (1, src_len=4, d_model=64)

        # 第二步：Decoder 自回归生成
        # 初始输入：只有 BOS
        generated = [BOS_ID]
        print(f"\n开始自回归生成：")
        print(f"  step 0: decoder 输入 = {generated}")

        for step in range(MAX_GEN_LEN):
            tgt_so_far = torch.tensor([generated])  # (1, current_len)

            # Decoder 一次性处理已生成序列，但因果掩码保证只看历史
            logits = model.decode(tgt_so_far, enc_output)
            # logits: (1, current_len, tgt_vocab_size)

            # 取最后一个位置的预测（当前步的输出）
            last_logits = logits[0, -1, :]  # (tgt_vocab_size,)
            next_token = last_logits.argmax(dim=-1).item()  # 贪心解码

            # 打印当前步的概率分布（前 5 个）
            probs = F.softmax(last_logits, dim=-1)
            top5 = probs.topk(5)
            top5_info = [(top5.indices[i].item(), f"{top5.values[i].item():.3f}")
                         for i in range(5)]
            print(f"  step {step + 1}: 预测 token = {next_token}  "
                  f"top-5 概率 = {top5_info}")

            generated.append(next_token)

            if next_token == EOS_ID:
                print(f"  （遇到 EOS，停止生成）")
                break

        print(f"\n生成结果: {generated}")

    # 验证训练时 forward 的输出 shape
    print(f"\n训练模式 forward 验证：")
    src_train = torch.randint(1, src_vocab_size, (2, 6))   # (batch=2, src_len=6)
    tgt_train = torch.randint(1, tgt_vocab_size, (2, 5))   # (batch=2, tgt_len=5)
    logits = model(src_train, tgt_train)
    print(f"  src shape:    {src_train.shape}")
    print(f"  tgt shape:    {tgt_train.shape}")
    print(f"  logits shape: {logits.shape}")
    assert logits.shape == (2, 5, tgt_vocab_size), \
        f"logits shape 不符合预期: {logits.shape}"
    print(f"  shape 验证通过")


def count_parameters():
    """统计完整 Transformer 的参数量"""
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
    )
    total = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    emb_params = (sum(p.numel() for p in model.src_embedding.parameters())
                  + sum(p.numel() for p in model.tgt_embedding.parameters()))
    proj_params = sum(p.numel() for p in model.output_projection.parameters())

    print(f"\n完整 Transformer 参数量（vocab=32000, d_model=512, N=6）：")
    print(f"  Embedding（src+tgt）: {emb_params:>12,}")
    print(f"  Encoder（6 层）:      {enc_params:>12,}")
    print(f"  Decoder（6 层）:      {dec_params:>12,}")
    print(f"  输出投影:             {proj_params:>12,}")
    print(f"  总计:                 {total:>12,}  ({total/1e6:.1f}M)")


if __name__ == "__main__":
    print("=" * 60)
    print("完整 Transformer 演示")
    print("=" * 60)

    demo_autoregressive_generation()
    count_parameters()
