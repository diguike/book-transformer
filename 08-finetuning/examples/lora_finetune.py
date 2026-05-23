"""
LoRA 微调演示：BERT-base 情感分类

运行要求：
  需要在有 GPU 的机器上运行。
  推荐实例：
    - 阿里云 ecs.gn6v-c8g1.2xlarge（NVIDIA V100 16GB）
    - 火山云 ml.g1e.xlarge（NVIDIA A10G 24GB）
  显存需求：约 8GB（bert-base-uncased + LoRA 微调，batch_size=32）

依赖安装：pip install -r requirements.txt
  版本要求：transformers>=4.38.0（4.38 起 evaluation_strategy 已更名为 eval_strategy）

任务：SST-2 二分类情感分析（正面/负面）
数据：训练集 1000 条，验证集 200 条（从 SST-2 随机采样）
"""

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── 环境检查 ─────────────────────────────────────────────────────────────────

print(f"PyTorch 版本：{torch.__version__}")
print(f"CUDA 可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备：{torch.cuda.get_device_name(0)}")
    # 微调开始前的显存基线
    print(f"初始显存占用：{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
print()

# ── 数据加载 ─────────────────────────────────────────────────────────────────

print("加载 SST-2 数据集...")
# SST-2 是 Stanford Sentiment Treebank 的二分类版本
# label: 1 = positive, 0 = negative
dataset = load_dataset("glue", "sst2")

# 从训练集抽取 1000 条，验证集抽取 200 条
# seed 固定确保实验可复现
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
val_dataset   = dataset["validation"].shuffle(seed=42).select(range(200))

print(f"训练集大小：{len(train_dataset)}")
print(f"验证集大小：{len(val_dataset)}")
print(f"样本示例：{train_dataset[0]}")
print()

# ── Tokenization ──────────────────────────────────────────────────────────────

MODEL_NAME = "bert-base-uncased"

# 加载 tokenizer，负责将原始文本转为 token ID
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """
    将 SST-2 的 sentence 字段 tokenize。
    max_length=128 对 SST-2 这类短文本已经足够，减少 padding 开销。
    truncation=True：超过 max_length 的文本会被截断
    padding="max_length"：不足的用 [PAD] 补齐
    """
    return tokenizer(
        examples["sentence"],
        max_length=128,
        truncation=True,
        padding="max_length",
    )

# batched=True 利用向量化批量处理，比逐条处理快 10 倍以上
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val   = val_dataset.map(tokenize_function, batched=True)

# 设置 PyTorch tensor 格式，Trainer 需要这个
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_val.set_format("torch",   columns=["input_ids", "attention_mask", "label"])

# ── 模型加载 ─────────────────────────────────────────────────────────────────

print(f"加载基础模型：{MODEL_NAME}")
# num_labels=2：在 BERT 顶部接一个输出维度为 2 的线性分类头
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

if torch.cuda.is_available():
    mem_after_load = torch.cuda.memory_allocated(0) / 1024**2
    print(f"模型加载后显存：{mem_after_load:.1f} MB")

# ── LoRA 配置 ─────────────────────────────────────────────────────────────────

# LoRA 核心配置说明：
#
# r (rank)：低秩矩阵的秩。决定 LoRA 的参数量。
#   r=8 是常见的起点：参数量适中，效果稳定。
#   增大 r（如 16、32）：可训练参数增多，表达能力更强，但收益递减。
#   减小 r（如 4）：进一步降低显存，适合显存极度受限的场景。
#
# lora_alpha：缩放系数。实际生效的缩放比例是 alpha/r。
#   惯例设置 alpha = 2 * r，这里 alpha=16, r=8，缩放比例为 2.0。
#   可以理解为控制 LoRA 更新对原始权重的"影响力"。
#
# target_modules：对哪些模块应用 LoRA。
#   BERT 的 attention 层有 query、key、value、dense 四个线性层。
#   通常只更新 query 和 value 就足够，覆盖更多模块可以提升效果但增加参数量。
#
# lora_dropout：应用在 LoRA 层上的 dropout，防止过拟合。
#   小数据集建议 0.1，大数据集可以降到 0.05。
#
# bias="none"：不对 bias 参数应用 LoRA，通常不影响效果。

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,      # 序列分类任务
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # 对 attention 的 Q、V 矩阵应用 LoRA
    lora_dropout=0.1,
    bias="none",
)

# 将 LoRA 层注入基础模型
# 这一步会冻结所有原始参数，只有 LoRA 新增的 A、B 矩阵是可训练的
model = get_peft_model(base_model, lora_config)

# 打印可训练参数统计，验证 LoRA 配置生效
model.print_trainable_parameters()
# 预期输出类似：
# trainable params: 296,450 || all params: 109,779,202 || trainable%: 0.27%

print()

# ── 评估指标 ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Trainer 在每个 eval_strategy 步骤结束后调用此函数。
    eval_pred 包含 (logits, labels) 两个 numpy array。
    """
    logits, labels = eval_pred
    # logits 形状：(batch_size, num_labels)
    # argmax 取概率最高的类别作为预测标签
    predictions = np.argmax(logits, axis=-1)

    # 整体准确率
    accuracy = (predictions == labels).mean()

    return {"accuracy": accuracy}

# ── 训练配置 ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,

    # batch_size 根据显存调整
    # V100 16GB 上 bert-base + LoRA 用 32 没问题
    # 如果显存不足，降到 16 或 8
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,

    # LoRA 微调的学习率通常比全量微调高，因为参数量少，需要更大步长
    learning_rate=2e-4,

    # 前 10% 步数做 warmup，线性从 0 增加到 learning_rate
    warmup_ratio=0.1,

    # 每个 epoch 结束后在验证集上评估一次
    eval_strategy="epoch",
    save_strategy="epoch",

    # 训练结束后自动加载验证集 loss 最低的 checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # 日志记录间隔（每 50 步打印一次训练 loss）
    logging_steps=50,

    # 只保留最近 2 个 checkpoint，节省磁盘空间
    save_total_limit=2,
)

# ── 训练 ─────────────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

print("开始训练...")
print(f"总训练步数：{len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
print()

trainer.train()

print()
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated(0) / 1024**2
    print(f"训练峰值显存：{peak_mem:.1f} MB")

# ── 详细评估 ──────────────────────────────────────────────────────────────────

print("\n在验证集上做详细评估...")

# 获取验证集的预测结果
predictions_output = trainer.predict(tokenized_val)
predicted_labels = np.argmax(predictions_output.predictions, axis=-1)
true_labels = predictions_output.label_ids

# 打印完整的分类报告
print("\n分类报告：")
print(classification_report(
    true_labels,
    predicted_labels,
    target_names=["negative", "positive"]
))

# ── 保存 LoRA 权重 ────────────────────────────────────────────────────────────

save_path = "./lora_weights"
model.save_pretrained(save_path)
print(f"LoRA 权重已保存到 {save_path}")
print("（只保存了 LoRA 参数，文件大小约 2-5MB，而不是完整模型的几百 MB）")

# ── 加载推理示例 ──────────────────────────────────────────────────────────────

print("\n演示：加载 LoRA 权重做推理...")
from peft import PeftModel

# 重新加载基础模型，然后挂载 LoRA 权重
inference_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
inference_model = PeftModel.from_pretrained(inference_model, save_path)
inference_model.eval()

if torch.cuda.is_available():
    inference_model = inference_model.cuda()

# 单条推理
test_texts = [
    "This movie is absolutely fantastic, I loved every moment!",
    "The film was boring and the plot made no sense at all.",
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = inference_model(**inputs)

    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    label_name = "positive" if predicted_class == 1 else "negative"

    print(f"  文本：{text[:50]}...")
    print(f"  预测：{label_name}\n")
