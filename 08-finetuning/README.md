---
title: 微调
feishu_url: "https://www.feishu.cn/wiki/IsM2wQhfwiGDkKk1lPzcB7E8nMq"
last_synced: "2026-05-07T23:07:13"
---

## 为什么通用模型不够用

预训练模型在大规模通用语料上训练，解决了通用语言理解问题。但在实际业务场景中，通用模型往往达不到生产要求，原因集中在三点。

**分布差异**。预训练语料以通用互联网文本为主，而业务数据有自己的风格和分布。法律合同、医疗病历、金融报告的文本模式与 Wikipedia 差距很大。在自己的业务数据上微调，可以让模型适应这种分布偏移。

**领域术语**。通用模型对专业术语的理解是基于上下文推断，而不是精确的领域定义。在法律 NLP 任务中，"条款"、"约定"、"义务"这类词有精确的法律含义；在医疗场景，同一个英文缩写可能在不同科室有完全不同的解释。微调可以让模型学到这种领域特定的语义。

**任务格式不匹配**。通用模型没有被训练成特定任务的输入输出格式。做分类任务，模型需要输出一个标签，而不是一段解释；做信息抽取，需要结构化 JSON，而不是自然语言描述。通过微调，可以让模型学会任务的输出格式约定。

当任务数据量较小（几百到几千条）但标注质量高时，微调通常比 prompt 工程更稳定，延迟也更低——微调后的模型推理时不需要附带大量的 few-shot 示例。

## 微调的本质

微调（fine-tuning）本质上是在预训练权重的基础上继续做梯度下降。预训练阶段学到的通用语言知识保留在权重里，微调用任务数据调整这些权重，使模型在目标任务上的损失降低。

从优化角度看，微调和预训练的区别只在于数据和学习率：微调用的数据是特定任务的标注数据，学习率通常比预训练低 1-2 个数量级（2e-5 到 5e-5），避免在小数据集上过拟合或破坏预训练学到的特征。

微调有三种常见粒度：

- **特征提取（Feature Extraction）**：冻结所有预训练层，只训练新增的分类头。速度最快，但效果往往不如更新模型权重。
- **全量微调（Full Fine-tuning）**：更新模型全部参数。效果最好，但显存需求随 batch size 增大而显著上升。BERT-base（1.1 亿参数）fp32 参数存储约 440MB，加上梯度和 Adam 优化器状态（m、v），纯参数侧约 1.7GB；激活值占用随 batch size 增大，batch_size=32、seq_len=512 时总显存约需 8-12GB，batch_size=8 时约需 4-6GB。
- **参数高效微调（PEFT）**：只更新极少数新增参数，保持预训练权重不变。LoRA 是目前最流行的 PEFT 方法。

## 全量微调 vs LoRA

### 全量微调

全量微调更新模型所有层的参数。优点是实现简单，理论上效果上限最高。缺点明显：

- 显存需求大。以 LLaMA-7B 为例，fp32 存储参数约 28GB，加上梯度和优化器状态（Adam 需要存储一阶和二阶矩），总显存消耗约 112GB，单张 A100 80GB 放不下，需要多卡并行。GPT-2 XL（1.5B 参数）全量微调约需 24GB 显存，也已超出消费级 GPU 的上限。
- 训练慢。所有参数都参与反向传播计算。
- 每个任务需要保存一份完整的模型权重，存储开销线性增长。

全量微调适用于：数据量充足（10 万条以上）、有足够的 GPU 资源、任务与预训练数据分布差距大。

### LoRA

LoRA（Low-Rank Adaptation）通过只训练极少数新增参数，大幅降低显存和计算开销。工程测试中，LoRA 可以将可训练参数量减少到全量微调的 0.1%-1%，显存节省 70% 以上，效果与全量微调接近。

**什么时候选 LoRA**：

- 显存有限（消费级 GPU 如 RTX 3090 24GB，或云服务器上的 A10G 24GB）
- 需要为多个任务维护不同版本的模型（基础模型共享，只保存各任务的 LoRA 权重）
- 数据量在数百到数万条之间
- 想快速验证微调效果

**什么时候选全量微调**：

- 拥有多卡高端 GPU（A100 80GB × 4 或以上）
- 任务对精度要求极高，且数据量充足
- 需要把预训练模型的领域知识大幅改写

## LoRA 原理

LoRA 的核心想法是：微调时模型权重的更新量 $\Delta W$ 实际上是低秩的——它不需要完整的高维矩阵来表达，而是可以用两个小矩阵的乘积近似。

对于原始权重矩阵 $W \in \mathbb{R}^{d \times k}$，LoRA 在旁边接入两个低秩矩阵：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

训练时，$W$ 冻结不动，只训练 $A$ 和 $B$。如果 $d=k=768$，$r=8$，那么：

- 原始权重参数量：768 × 768 = 589,824
- LoRA 参数量：768×8 + 8×768 = 12,288
- 参数量是原来的 **2.1%**

推理时把 $\Delta W = BA$ 合并回 $W$，不增加额外的推理延迟。

几个关键配置参数：

- **`r`（rank）**：低秩矩阵的秩，常用值 4、8、16。r 越大，表达能力越强，显存和参数量也随之增加。默认从 8 开始尝试。
- **`lora_alpha`**：缩放系数，控制 $\Delta W$ 对原始权重的影响幅度。实际缩放比例是 `lora_alpha / r`，常见设置是 `alpha = 2 * r`。
- **`target_modules`**：指定对哪些模块应用 LoRA。BERT 和 GPT 类模型通常对 attention 层的 Q、V 矩阵应用 LoRA，即 `["query", "value"]`。也可以包含 K 和输出矩阵，覆盖范围越广效果越好，参数量也越多。

## 数据格式

### 分类任务

分类任务的标注数据格式简单，每条样本包含文本和标签：

```json
{"text": "This product is absolutely fantastic!", "label": 1}
{"text": "Terrible experience, will not buy again.", "label": 0}
```

标签用整数，而不是字符串。HuggingFace Datasets 的 `ClassLabel` 特征会自动处理标签到整数的映射。

分类任务的模型在 `[CLS]` token 上接一个线性层输出 logits，训练目标是交叉熵损失。

### 指令微调

指令微调（Instruction Fine-tuning）用于对话和生成模型，数据格式是输入-输出对：

```json
{
  "instruction": "将下面的句子翻译成英文",
  "input": "今天天气很好，适合出去走走。",
  "output": "The weather is nice today, it's a good time to go for a walk."
}
```

也有更简洁的 Alpaca 格式，将 instruction 和 input 合并成一个 prompt：

```json
{
  "prompt": "### 指令\n将下面的句子翻译成英文\n\n### 输入\n今天天气很好，适合出去走走。\n\n### 输出\n",
  "completion": "The weather is nice today, it's a good time to go for a walk."
}
```

### 数据质量 vs 数据量

一个常见误区是堆砌数据量。实际上，1000 条高质量、格式统一的数据，通常比 10 万条含噪声的数据效果更好。数据质量问题包括：标注不一致、输出格式混乱、文本截断、重复样本。

数据质量对微调效果的影响超过大多数超参数。1000 条标注准确、格式统一的数据，通常胜过 10000 条含噪声的数据——这已被大量实验验证。

## 云服务器配置

微调工作需要 GPU 支持。以下是在阿里云和火山云上的推荐配置。

### 阿里云

推荐实例系列：**ecs.gn6v**（搭载 NVIDIA V100 16GB）或 **ecs.gn7i**（搭载 A10 24GB）

| 实例规格 | GPU | 显存 | 适用场景 | 参考价格（按量付费） |
|---------|-----|------|---------|-----------------|
| ecs.gn6v-c8g1.2xlarge | V100 × 1 | 16GB | BERT 级别全量微调，LoRA 微调 7B 以下模型 | ~¥20/小时（以上为写作时参考价，请以官网实时报价为准） |
| ecs.gn7i-c16g1.4xlarge | A10 × 1 | 24GB | 全量微调 1B 以下，LoRA 微调 13B 以下 | ~¥15/小时（以上为写作时参考价，请以官网实时报价为准） |
| ecs.gn7i-c32g1.8xlarge | A10 × 2 | 48GB | 多卡训练，大模型全量微调 | ~¥30/小时（以上为写作时参考价，请以官网实时报价为准） |

### 火山云

推荐系列：**ml.g1e**（搭载 A10G）或 **ml.g2**（搭载 A100）

| 实例规格 | GPU | 显存 | 参考价格 |
|---------|-----|------|---------|
| ml.g1e.xlarge | A10G × 1 | 24GB | ~¥12/小时（以上为写作时参考价，请以官网实时报价为准） |
| ml.g2.large   | A100 × 1 | 80GB | ~¥30/小时（以上为写作时参考价，请以官网实时报价为准） |

### 费用估算

以本章代码示例（BERT-base LoRA 微调，SST-2 数据集子集 1000 条，3 个 epoch）为例：

- 实例：ecs.gn6v-c8g1.2xlarge（V100 16GB）
- 训练时间：约 15-20 分钟
- 费用：约 ¥5-7

正式业务的微调任务（GPT-2 全量微调，10 万条数据，10 epoch）大约需要 2-4 小时，费用约 ¥40-80。

### 环境准备

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

HuggingFace 模型下载在国内可能较慢，建议配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 微调流程

用 HuggingFace `Trainer` 配合 `PEFT` 库做 LoRA 微调，标准流程如下：

```
原始数据
    ↓ 数据清洗、格式化
HuggingFace Dataset
    ↓ tokenize（map + batched）
tokenized Dataset
    ↓
预训练模型 + LoRA 配置
    ↓ get_peft_model()
PEFT Model（只有 LoRA 参数可训练）
    ↓
TrainingArguments（超参数配置）
    ↓
Trainer.train()
    ↓
评估 + 保存 LoRA 权重
```

关键步骤说明：

**1. 加载基础模型**

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

**2. 配置 LoRA**

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出类似：trainable params: 296,450 || all params: 109,779,202 || trainable%: 0.27
```

**3. 定义 TrainingArguments**

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,        # LoRA 微调学习率通常比全量微调高一些
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

**4. 训练和保存**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
trainer.train()

# 只保存 LoRA 权重（几 MB），不是完整模型（几百 MB）
model.save_pretrained("./lora_weights")
```

## 评估

### Loss 曲线

训练时关注两条 loss 曲线：训练 loss 和验证 loss。正常情况下两条曲线都应下降，且验证 loss 略高于训练 loss。

过拟合的典型表现：训练 loss 持续下降，但验证 loss 在某个 epoch 后开始上升。出现这种情况应该用此 epoch 的 checkpoint 而不是最终的权重，所以 `load_best_model_at_end=True` 很重要。

常见处理方式：
- 减少训练轮次（3-5 epoch 通常足够）
- 增大 dropout（`lora_dropout` 从 0.05 调到 0.1-0.2）
- 数据增强
- 减小 LoRA 的 `r`

### Classification Report

分类任务用 `sklearn.metrics.classification_report` 输出每个类别的 precision、recall、F1：

```python
from sklearn.metrics import classification_report

report = classification_report(true_labels, predicted_labels, target_names=["negative", "positive"])
print(report)
```

输出示例：

```
              precision    recall  f1-score   support

    negative       0.92      0.91      0.91       100
    positive       0.91      0.92      0.91       100

    accuracy                           0.91       200
   macro avg       0.91      0.91      0.91       200
weighted avg       0.91      0.91      0.91       200
```

accuracy 是整体指标，但在类别不平衡时容易误导，建议同时看各类别的 F1。

### 避免过拟合的实践

- 数据集越小，越容易过拟合，epoch 数要设小（1-3）
- 学习率用 warmup，不要一开始就用最大学习率
- 验证集比例保持 10-20%，且分布与训练集一致
- 如果有条件，使用早停（early stopping）：`EarlyStoppingCallback`
