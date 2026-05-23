"""
4-bit 量化推理演示

运行要求：
  需要 CUDA GPU（bitsandbytes 量化不支持 CPU）。
  推荐实例：
    - 阿里云 ecs.gn6v-c8g1.2xlarge（V100 16GB）
    - 火山云 ml.g1e.xlarge（A10G 24GB）
  显存需求：约 2GB（facebook/opt-125m 是小模型，量化前约 500MB，用于教学演示）

依赖安装：pip install -r requirements.txt

演示内容：
  1. 对比量化前后模型的显存占用
  2. 做一次前向推理，验证量化后输出正常
  3. 说明 load_in_4bit 背后的机制
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── 环境检查 ─────────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    raise RuntimeError(
        "bitsandbytes 量化需要 CUDA GPU。\n"
        "请在阿里云 ecs.gn6v 或火山云 ml.g1e 实例上运行此脚本。"
    )

print(f"GPU 设备：{torch.cuda.get_device_name(0)}")
print(f"GPU 总显存：{torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
print()

MODEL_NAME = "facebook/opt-125m"
# opt-125m 是 Meta 发布的小型 GPT 风格模型，1.25 亿参数，约 500MB
# 适合演示量化效果，显存需求低，下载快

# ── 未量化版本（fp16） ────────────────────────────────────────────────────────

print("=" * 60)
print("加载 fp16 模型（未量化）...")

# 重置显存统计，确保测量准确
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# 记录加载前的显存基线
mem_before = torch.cuda.memory_allocated() / 1024**2

model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

torch.cuda.synchronize()
mem_fp16 = torch.cuda.memory_allocated() / 1024**2 - mem_before

print(f"fp16 模型显存占用：{mem_fp16:.1f} MB")

# 做一次推理验证模型正常
test_input = "The transformer architecture was introduced in"
inputs = tokenizer(test_input, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_fp16 = model_fp16.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,          # 贪心解码，输出确定性，便于对比
    )

fp16_text = tokenizer.decode(output_fp16[0], skip_special_tokens=True)
print(f"fp16 推理输出：{fp16_text}")

# 释放 fp16 模型，腾出显存
del model_fp16
torch.cuda.empty_cache()
print()

# ── 4-bit 量化版本 ────────────────────────────────────────────────────────────

print("=" * 60)
print("加载 4-bit 量化模型...")

# load_in_4bit=True 背后做了什么：
#
# 1. 权重量化：将模型权重从 fp16（16 bit）压缩到 NF4（4 bit）格式。
#    NF4（Normal Float 4）是专为正态分布权重设计的 4-bit 格式，
#    比普通 fp4 精度更高，因为神经网络权重通常近似正态分布。
#
# 2. 双重量化（bnb_4bit_use_double_quant=True）：
#    对量化系数（scale）再做一次量化，额外节省约 0.5 bit/参数的显存。
#
# 3. 计算时反量化（bnb_4bit_compute_dtype=float16）：
#    权重以 4-bit 存储，但矩阵乘法前临时反量化为 fp16 再计算。
#    这样保留了计算精度，同时显存占用按 4-bit 计算。
#    这种方式称为 "weight-only quantization"（仅权重量化）。
#
# 4. 对 Linear 层替换：bitsandbytes 把模型中的 nn.Linear 替换为
#    自定义的 Linear4bit 层，内部处理量化/反量化，对外接口不变。

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated() / 1024**2

model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

torch.cuda.synchronize()
mem_4bit = torch.cuda.memory_allocated() / 1024**2 - mem_before

print(f"4-bit 量化模型显存占用：{mem_4bit:.1f} MB")

# ── 显存对比 ─────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("显存对比：")
print(f"  fp16（未量化）：{mem_fp16:.1f} MB")
print(f"  4-bit 量化：   {mem_4bit:.1f} MB")
if mem_fp16 > 0:
    reduction = (1 - mem_4bit / mem_fp16) * 100
    print(f"  显存节省：     {reduction:.1f}%")
print()

# ── 量化后推理验证 ─────────────────────────────────────────────────────────────

print("=" * 60)
print("验证量化后推理输出...")

inputs = tokenizer(test_input, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_4bit = model_4bit.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

text_4bit = tokenizer.decode(output_4bit[0], skip_special_tokens=True)
print(f"4-bit 推理输出：{text_4bit}")
print()

# ── 更多推理示例 ──────────────────────────────────────────────────────────────

print("=" * 60)
print("更多推理示例：")

prompts = [
    "Artificial intelligence is",
    "Python is a programming language that",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_4bit.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n  输入：{prompt}")
    print(f"  输出：{generated}")

print()
print("=" * 60)
print("结论：4-bit 量化在显著减少显存的同时，输出质量基本保持。")
print("对于 125M 这类小模型效果演示意义为主；大模型（7B+）量化收益更明显。")
