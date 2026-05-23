"""
三种 Transformer 架构对比示例

演示：
1. 加载 BERT（Encoder-Only）、GPT-2（Decoder-Only）、T5-small（Encoder-Decoder）
2. 打印每个模型的参数量
3. 演示每种模型的基本推理用法

运行前安装依赖：
    pip install -r requirements.txt

注意：首次运行会从 HuggingFace Hub 下载模型权重，需要网络连接。
模型会缓存在 ~/.cache/huggingface/ 下，后续运行不再下载。
"""

import torch
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,
)


def count_parameters(model) -> int:
    """计算模型的可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_param_count(n: int) -> str:
    """将参数量格式化为易读的字符串，如 110.1M"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    return f"{n / 1_000_000:.1f}M"


# ─────────────────────────────────────────────
# 1. BERT（Encoder-Only）
#    适合任务：文本分类、NER、语义相似度、抽取式问答
#    特点：双向注意力，输出每个 token 的上下文向量
# ─────────────────────────────────────────────

print("=" * 60)
print("BERT（Encoder-Only）")
print("适合：文本分类、NER、语义匹配、抽取式问答")
print("=" * 60)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# BertModel 输出 token 级别的隐层向量，不包含分类头
# 如果需要分类任务，用 BertForSequenceClassification
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

print(f"参数量：{format_param_count(count_parameters(bert_model))}")

# 演示：获取句子的 [CLS] 向量（通常用于句子级别的表示）
text = "The bank can guarantee deposits will eventually cover future tuition costs."
inputs = bert_tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = bert_model(**inputs)

# last_hidden_state: [batch_size, seq_len, hidden_size]
last_hidden_state = outputs.last_hidden_state
# [CLS] token 的向量，形状 [batch_size, hidden_size]
cls_vector = last_hidden_state[:, 0, :]

print(f"输入 token 数：{inputs['input_ids'].shape[1]}")
print(f"每个 token 的隐层维度：{last_hidden_state.shape[-1]}")
print(f"[CLS] 向量形状（句子表示）：{cls_vector.shape}")
# 实际使用中，cls_vector 接一个线性分类层即可做句子分类
print()


# ─────────────────────────────────────────────
# 2. GPT-2（Decoder-Only）
#    适合任务：文本生成、对话、续写、指令跟随
#    特点：因果注意力（只看左侧），自回归生成
# ─────────────────────────────────────────────

print("=" * 60)
print("GPT-2（Decoder-Only）")
print("适合：文本生成、续写、对话、指令跟随")
print("=" * 60)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT2LMHeadModel 包含语言模型头，可以直接生成文本
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

print(f"参数量：{format_param_count(count_parameters(gpt2_model))}")

# 演示：给定 prompt，自回归生成后续文本
prompt = "Artificial intelligence is transforming"
inputs = gpt2_tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    # max_new_tokens：最多生成多少个新 token
    # do_sample=False：贪心解码，每步选概率最高的 token（结果确定）
    output_ids = gpt2_model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=gpt2_tokenizer.eos_token_id,
    )

generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"输入 prompt：{prompt}")
print(f"生成结果：{generated_text}")
print()


# ─────────────────────────────────────────────
# 3. T5-small（Encoder-Decoder）
#    适合任务：翻译、摘要、生成式问答、文本纠错
#    特点：Encoder 双向理解输入，Decoder 生成输出
#    T5 把所有任务统一为"文本到文本"格式
# ─────────────────────────────────────────────

print("=" * 60)
print("T5-small（Encoder-Decoder）")
print("适合：翻译、摘要、生成式问答、文本纠错")
print("=" * 60)

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.eval()

print(f"参数量：{format_param_count(count_parameters(t5_model))}")

# 演示：T5 的任务格式是在输入前加任务前缀
# T5-small 是基础预训练模型，翻译效果有限，这里主要演示接口用法
task_input = "translate English to French: How are you doing today?"
inputs = t5_tokenizer(task_input, return_tensors="pt")

with torch.no_grad():
    output_ids = t5_model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=4,          # 束搜索，比贪心解码质量更好
        early_stopping=True,  # 生成 EOS token 时停止
    )

output_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"输入（任务前缀 + 内容）：{task_input}")
print(f"输出：{output_text}")
print()


# ─────────────────────────────────────────────
# 参数量汇总
# ─────────────────────────────────────────────

print("=" * 60)
print("参数量汇总")
print("=" * 60)
print(f"BERT-base-uncased  (Encoder-Only)    : {format_param_count(count_parameters(bert_model))}")
print(f"GPT-2              (Decoder-Only)    : {format_param_count(count_parameters(gpt2_model))}")
print(f"T5-small           (Encoder-Decoder) : {format_param_count(count_parameters(t5_model))}")
print()
print("说明：")
print("- 参数量是模型规模的基本指标，但不直接等于任务性能")
print("- BERT-large: 340M, GPT-2 XL: 1.5B, T5-11B: 11B")
print("- 现代大模型（LLaMA-3-70B）是这里 GPT-2 的数万倍")
