---
title: 推理工程
feishu_url: "https://www.feishu.cn/wiki/R7few5piQiRM8kkGMsIcxOC5n5f"
last_synced: "2026-05-07T23:07:20"
---

## 推理的工程挑战

把一个训练好的模型部署到生产环境，核心关注三个指标：**延迟（Latency）**、**吞吐量（Throughput）**、**显存（VRAM）**。

**延迟**是从请求到达到第一个 token（或完整结果）返回的时间，对用户体验直接可见。对话场景通常要求 P99 延迟在 1 秒以内，首 token 延迟（TTFT, Time to First Token）在 500ms 以内。

**吞吐量**是单位时间内能处理的 token 数量（tokens/second）或请求数量（QPS）。高吞吐量意味着更低的单请求成本，对 ToB 服务尤其重要。

**显存**决定了能部署多大的模型，以及批处理的上限。显存不足会触发 OOM（Out of Memory）导致服务崩溃，或强迫使用更小的 batch size，降低吞吐量。

三者之间存在权衡：

- 增大 batch size → 吞吐量上升，但延迟也上升（每个请求要等攒够一批才处理）
- 量化到低精度 → 显存减少，吞吐量通常上升，但精度略有损失
- 增加并发流 → 吞吐量上升，但竞争显存，可能反而增加延迟

工程上不存在"最优解"，需要根据具体业务 SLA 和硬件条件做权衡。

## 模型加载与显存管理

### 精度格式

模型权重有几种精度格式，影响显存占用和计算速度：

| 格式 | 位宽 | 显存（以 1B 参数为例） | 特点 |
|------|------|-------------------|------|
| fp32 | 32位浮点 | 4GB | 训练默认，精度最高 |
| fp16 | 16位浮点 | 2GB | 推理常用，大多数 GPU 支持 |
| bf16 | 16位脑浮点 | 2GB | 更大的指数范围，A100/H100 原生支持，训练更稳定 |
| INT8 | 8位整数 | 1GB | 量化，精度小幅下降 |
| INT4 | 4位整数 | 0.5GB | 激进量化，精度损失明显但通常可接受 |

`fp16` 和 `bf16` 的显存占用相同，区别在数值范围：`fp16` 数值范围较小，大模型推理时容易出现数值溢出；`bf16` 的指数位更多，数值范围与 `fp32` 相同，在 Ampere 架构以上的 GPU（A100、RTX 3090）上是更好的选择。

加载时通过 `torch_dtype` 指定精度：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,   # 或 torch.bfloat16
)
```

### device_map="auto"

`device_map="auto"` 是 `accelerate` 库提供的自动设备分配功能。它分析模型各层的参数量，按显存大小把模型层分配到可用的设备上，支持多 GPU 甚至 CPU-offload（把放不下的层卸载到内存）。

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",           # 自动分配到可用 GPU，不够则溢出到 CPU
)
```

`device_map="auto"` 适合快速验证阶段。生产部署时建议明确指定 `device_map` 的分配策略，或直接使用 vLLM 等专业推理框架，避免跨设备通信带来的性能损耗。

### 多 GPU 推理

两种方式：

**模型并行（Model Parallelism）**：把模型的不同层分配到不同 GPU，`device_map="auto"` 默认就是这种方式。适合单个模型放不进一张显卡的场景，但 GPU 间通信有开销。

**张量并行（Tensor Parallelism）**：把同一层的权重矩阵按列或行切分到多 GPU，需要专门的框架支持（vLLM、Megatron-LM）。通信开销更大，但计算并行度更高，适合追求极致吞吐量的场景。

## 量化

### 原理简介

量化（Quantization）把浮点数权重压缩成低比特整数表示。以 INT8 量化为例：

```
浮点权重值 w ∈ [w_min, w_max]
映射到整数 q ∈ [-128, 127]

量化：q = round(w / scale + zero_point)
反量化：w ≈ scale * (q - zero_point)
```

`scale` 和 `zero_point` 是量化参数，每个 tensor 或每行/列存储一套，推理时先反量化再做矩阵乘法（Weight-only quantization），或直接用整数运算（全整数量化）。

精度损失来源于量化误差（`w` 和反量化后的近似值之差），通常在 0.5% 以内，大多数任务可以接受。

### bitsandbytes 4-bit 量化

`bitsandbytes` 是目前最易用的量化库，支持 INT8 和 NF4（4-bit Normal Float）量化：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# load_in_4bit=True 背后做了什么：
#   1. 将模型权重从 fp16/fp32 量化为 NF4 格式（4 bit）
#   2. 计算时反量化为 fp16 再做矩阵乘法（compute_dtype）
#   3. 激活值仍然以 fp16 存储
#   整体效果：显存减少约 75%，推理速度接近 fp16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 计算时临时转回 fp16
    bnb_4bit_use_double_quant=True,        # 对量化参数再做一次量化，进一步省显存
    bnb_4bit_quant_type="nf4",             # NF4 比 fp4 精度更高
)

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    quantization_config=bnb_config,
    device_map="auto",
)
```

注意：`bitsandbytes` 的量化功能需要 CUDA，CPU-only 环境无法使用。

### 量化对精度的影响

在主流基准测试（MMLU、HellaSwag）上，4-bit 量化通常导致 1-3% 的性能下降，对大多数应用场景可以接受。

精度损失与模型大小有关：参数量越大的模型，量化对精度的影响越小。7B 参数模型 4-bit 量化的精度损失远小于 1B 参数模型。这是因为大模型有更强的"冗余"，低精度表示仍然能捕获主要信息。

## KV Cache

### 为什么需要 KV Cache

Decoder 模型（GPT、LLaMA 等）生成文本是逐 token 的自回归过程：每次前向传播只生成一个新 token，然后把这个 token 拼到输入末尾，再做下一次前向传播。

问题在于，每次前向传播都要重新计算所有已生成 token 的 K（Key）和 V（Value）矩阵。如果已生成 100 个 token，第 101 次前向传播就要做 100 次"没有新信息的重复计算"。

### KV Cache 如何减少重复计算

KV Cache 的解决方案是：把每次前向传播中计算出的 K 和 V 矩阵缓存起来，下次只计算新 token 的 K 和 V，再和缓存的历史 K、V 拼接后做 attention。

```
无缓存（第 n+1 步）：计算 n+1 个 token 的 K、V，做全量 attention
有缓存（第 n+1 步）：只计算第 n+1 个 token 的 K、V，与缓存的 n 组 K、V 拼接后做 attention
```

KV Cache 避免了对历史 token 重复计算 K、V 的线性投影。无缓存时，生成第 t 步需要对所有 t 个 token 重新计算一遍 K 和 V；有了缓存，每步只计算新 token 的 K/V，历史 token 的结果从缓存中直接读取。生成 n 个 token 的 KV 投影总次数从 $O(n^2)$ 降为 $O(n)$，长序列生成时提速显著。Attention 点积计算本身（Query 与所有历史 Key 做相似度）仍是 $O(n)$（每步），但这部分开销通常小于 KV 投影。

代价是显存：每个 token、每个 attention 头都需要存储 K 和 V，KV Cache 的显存占用随序列长度线性增长。以 LLaMA-7B 为例，生成 2048 token 的 KV Cache 约占用 1GB 显存。

HuggingFace 的 `generate()` 默认启用 KV Cache，通过 `use_cache=True`（默认值）控制。

## 批处理

### Dynamic Batching

服务端接收到请求后，不必等每个请求都有结果才返回，可以把多个并发请求打包成一个 batch 一起处理，提高 GPU 利用率。这就是 Dynamic Batching。

静态 batching 要求 batch 内所有序列长度相同，不够则 padding 到最长。Dynamic batching 则是随时把可以合批的请求打包，无需等待固定 batch size 凑满。

### Padding 和 Attention Mask

批量推理时，不同长度的序列需要 padding 到相同长度。padding token 不应该参与 attention 计算，这由 `attention_mask` 控制：

```python
# attention_mask: 1 表示真实 token，0 表示 padding token
# tokenizer 会自动生成
inputs = tokenizer(
    texts,
    padding=True,          # 自动 padding 到 batch 内最长序列
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

# inputs["attention_mask"] 的形状：(batch_size, seq_len)
# 值为 1 的位置参与 attention，值为 0 的位置被 mask 掉
outputs = model(**inputs)
```

padding 位置的选择也有讲究：

- **右 padding**（默认）：在序列末尾补零，适合 encoder 模型（BERT）
- **左 padding**：在序列开头补零，适合 decoder 模型（GPT）——因为生成时从序列末尾开始，右侧补零会导致生成错误

## 生产级推理服务

### vLLM：PagedAttention

vLLM 是目前生产环境最常用的 LLM 推理框架，核心创新是 **PagedAttention**。

传统推理框架为每个请求预分配一整块连续的 KV Cache 显存（按最大序列长度），导致大量碎片化浪费。PagedAttention 借鉴操作系统的虚拟内存管理思想：把 KV Cache 分成固定大小的"页"（page），按需分配，用页表管理物理显存到逻辑序列的映射。结果是显存利用率从约 60% 提升到 90% 以上，吞吐量大幅提升。

vLLM 的使用方式：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Tell me about transformers."], sampling_params)
print(outputs[0].outputs[0].text)
```

vLLM 还支持兼容 OpenAI API 的服务端模式：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

启动后就可以用 OpenAI SDK 直接访问，迁移成本极低。

### Ollama：本地部署 LLM

Ollama 是面向本地部署场景的工具，安装简单，支持 Mac/Linux/Windows：

```bash
# 安装 Ollama（Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 下载并运行模型（自动处理量化和显存分配）
ollama run llama3

# Python 调用
import ollama
response = ollama.chat(model='llama3', messages=[
    {'role': 'user', 'content': 'What is a transformer model?'}
])
print(response['message']['content'])
```

Ollama 自动做 4-bit 量化，7B 模型在 16GB 内存的 MacBook 上就能流畅运行。

### 选型建议

| 场景 | 推荐方案 |
|------|---------|
| 本地开发、原型验证 | Ollama |
| 生产服务、高并发 | vLLM |
| 中等规模、快速上线 | HuggingFace TGI（Text Generation Inference）|
| 只需要 embedding | sentence-transformers + FastAPI |

## 性能调优工具

### 测量延迟和吞吐量

```python
import time
import torch

def benchmark(model, inputs, num_runs=100, warmup=10):
    """
    测量模型推理的延迟和吞吐量。
    
    warmup：前几次推理用于 JIT 编译和 GPU 预热，不计入统计。
    """
    # GPU 操作是异步的，需要 synchronize() 确保计时准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 预热
    for _ in range(warmup):
        with torch.no_grad():
            model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 正式测量
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # 转换为毫秒

    import numpy as np
    return {
        "mean_ms": np.mean(latencies),
        "p50_ms":  np.percentile(latencies, 50),
        "p99_ms":  np.percentile(latencies, 99),
    }
```

测量时常见的坑：

- **忘记 `torch.cuda.synchronize()`**：GPU 操作异步，不同步会导致计时提前结束，测出来的延迟偏低
- **没有 warmup**：第一次推理包含 JIT 编译时间，不代表稳态性能
- **只测平均延迟**：P99 延迟才是用户体验的真实上界，尤其在有 GC 的 Python 环境中

### 显存分析

```python
# 查看当前显存占用
print(f"已分配显存：{torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"缓存显存：  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

# 峰值显存（从进程启动以来的最大值）
print(f"峰值显存：  {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

# 重置峰值统计（用于对比不同阶段的显存峰值）
torch.cuda.reset_peak_memory_stats()
```

PyTorch Profiler 可以给出更细粒度的 op-level 分析，适合定位具体的显存瓶颈：

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
    with torch.no_grad():
        model(**inputs)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

---

> 本章来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
