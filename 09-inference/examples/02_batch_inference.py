"""
批量推理演示：padding、attention_mask 与性能对比

运行环境：CPU 或 GPU 均可（GPU 效果对比更明显）
依赖安装：pip install -r requirements.txt

演示内容：
  1. 正确使用 padding 和 attention_mask 做批量推理
  2. 演示 decoder 模型需要左 padding 的原因
  3. 对比单条 vs 批量推理的时间
"""

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 模型加载 ─────────────────────────────────────────────────────────────────

MODEL_NAME = "facebook/opt-125m"

print(f"加载模型：{MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"运行设备：{device}")
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model = model.to(device)
model.eval()

# ── Padding 与 Attention Mask 说明 ────────────────────────────────────────────

print("=" * 60)
print("1. Padding 与 Attention Mask 演示")
print()

# 不同长度的文本（故意设计长度差异大）
sample_texts = [
    "Hello",
    "The weather is nice today",
    "Transformers are powerful neural network architectures used in modern NLP",
]

# tokenizer 的 padding 设置
# padding=True：自动 padding 到 batch 内最长序列（动态 padding，比 max_length 更高效）
# 等价于 padding="longest"
inputs = tokenizer(
    sample_texts,
    padding=True,       # 自动 padding 到 batch 内最长
    return_tensors="pt",
    truncation=True,
    max_length=128,
)

print("原始文本长度（token 数）：")
for i, text in enumerate(sample_texts):
    single_tokens = tokenizer(text, return_tensors="pt")
    print(f"  [{i}] {len(single_tokens['input_ids'][0])} tokens: {text[:40]}")

print(f"\npadding 后的 input_ids 形状：{inputs['input_ids'].shape}")
print(f"（所有序列都 padding 到最长序列的长度）")
print()

# attention_mask 的结构
# 1 = 真实 token，0 = padding token（不参与 attention 计算）
print("attention_mask 示例（1=真实token，0=padding）：")
for i in range(len(sample_texts)):
    mask = inputs['attention_mask'][i].tolist()
    ones = sum(mask)
    zeros = len(mask) - ones
    print(f"  [{i}] 真实 token: {ones}，padding token: {zeros}")
    # 打印前 20 个 mask 值
    preview = mask[:20]
    print(f"       mask 前缀：{preview}{'...' if len(mask) > 20 else ''}")

print()

# ── 左 Padding vs 右 Padding ──────────────────────────────────────────────────

print("=" * 60)
print("2. Decoder 模型需要左 Padding")
print()

# Decoder 模型（GPT、OPT 等）生成文本时，新 token 接在序列末尾。
# 如果使用右 padding（在末尾补零），padding token 会插入到生成位置之前，
# 导致模型"从 padding token 开始生成"，输出错误。
#
# 左 padding（在序列开头补零）才是正确做法：
# 真实 token 连续排在末尾，生成从正确位置开始。
#
# 对比：
#   右 padding：[BOS] hello world [PAD] [PAD] → 生成从 [PAD] 后开始（错误）
#   左 padding：[PAD] [PAD] [BOS] hello world → 生成从 world 后开始（正确）

# OPT 的 tokenizer 默认左 padding
print(f"OPT tokenizer 的 padding_side：{tokenizer.padding_side}")
# 如果不是 left，手动设置：
tokenizer.padding_side = "left"
print("已确认设置为左 padding（left padding）")
print()

# ── 批量推理示例（batch_size=8） ──────────────────────────────────────────────

print("=" * 60)
print("3. 批量推理（batch_size=8）")
print()

# 8 条不同长度的输入文本
texts_batch = [
    "The capital of France is",
    "In machine learning, overfitting occurs when",
    "Python was created by",
    "The largest planet in our solar system is",
    "Neural networks are inspired by",
    "The speed of light is approximately",
    "Deep learning models require",
    "The invention of the internet",
]

print("输入文本：")
for i, t in enumerate(texts_batch):
    print(f"  [{i}] {t}")
print()

# tokenize，左 padding 确保 decoder 正确生成
batch_inputs = tokenizer(
    texts_batch,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

print(f"Batch input_ids 形状：{batch_inputs['input_ids'].shape}")
print(f"attention_mask 形状：{batch_inputs['attention_mask'].shape}")
print()

with torch.no_grad():
    outputs = model.generate(
        **batch_inputs,
        max_new_tokens=15,
        do_sample=False,    # 贪心解码，确保输出可复现
        pad_token_id=tokenizer.pad_token_id,
    )

print("批量推理输出：")
for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"  [{i}] {generated_text}")

print()

# ── 单条 vs 批量推理时间对比 ──────────────────────────────────────────────────

print("=" * 60)
print("4. 单条 vs 批量推理时间对比")
print()

NUM_SAMPLES = 8    # 对比样本数
MAX_NEW_TOKENS = 20
WARMUP_RUNS = 3    # GPU 预热次数

# 确保模型在正确设备上
if device == "cuda":
    torch.cuda.synchronize()

# 预热：避免第一次推理包含 CUDA 内核编译开销
print("预热中...")
warmup_inputs = tokenizer(texts_batch[:2], padding=True, return_tensors="pt")
warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
for _ in range(WARMUP_RUNS):
    with torch.no_grad():
        model.generate(**warmup_inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)

if device == "cuda":
    torch.cuda.synchronize()

# ── 单条推理：逐条处理 8 个样本 ──
single_latencies = []

for text in texts_batch:
    single_input = tokenizer(text, return_tensors="pt")
    single_input = {k: v.to(device) for k, v in single_input.items()}

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        model.generate(
            **single_input,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    single_latencies.append((t_end - t_start) * 1000)

total_single_time = sum(single_latencies)

# ── 批量推理：一次处理全部 8 个样本 ──
all_inputs = tokenizer(
    texts_batch,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)
all_inputs = {k: v.to(device) for k, v in all_inputs.items()}

if device == "cuda":
    torch.cuda.synchronize()
t_batch_start = time.perf_counter()

with torch.no_grad():
    model.generate(
        **all_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

if device == "cuda":
    torch.cuda.synchronize()
t_batch_end = time.perf_counter()
total_batch_time = (t_batch_end - t_batch_start) * 1000

# ── 打印对比结果 ──
print(f"样本数量：{NUM_SAMPLES} 条，每条生成 {MAX_NEW_TOKENS} 个新 token")
print()
print(f"单条推理（逐条处理）：")
print(f"  各条耗时：{[f'{t:.0f}ms' for t in single_latencies]}")
print(f"  总耗时：  {total_single_time:.1f} ms")
print()
print(f"批量推理（一次处理全部）：")
print(f"  总耗时：  {total_batch_time:.1f} ms")
print()

if total_single_time > 0:
    speedup = total_single_time / total_batch_time
    print(f"批量推理加速比：{speedup:.2f}x")
    print()

# 解释说明
print("结论：")
if device == "cuda":
    print("  GPU 并行计算能力强，批量推理的加速比通常在 2-8x 之间。")
    print("  batch_size 越大，GPU 利用率越高，加速越明显。")
    print("  但 batch_size 也受限于显存，需要根据模型大小和序列长度调整。")
else:
    print("  CPU 环境下批量推理的加速比相对较小。")
    print("  在 GPU 上运行此脚本可以看到更显著的加速效果（通常 3-8x）。")

print()
print("=" * 60)
print("关键要点：")
print("  1. 批量推理时，padding=True 自动对齐序列长度")
print("  2. attention_mask 告诉模型哪些位置是真实 token，哪些是 padding")
print("  3. Decoder 模型必须使用左 padding（padding_side='left'）")
print("  4. pad_token_id 必须明确传入 generate()，避免警告和潜在错误")
