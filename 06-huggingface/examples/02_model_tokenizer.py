"""
Model + Tokenizer：精细控制

演示用 bert-base-uncased 做文本分类的完整流程：
    tokenize → model forward → logits → softmax → 预测类别

并与 pipeline 对比，展示 pipeline 封装了哪些步骤。

运行前安装依赖：
    pip install -r requirements.txt

注意：bert-base-uncased 是未经分类任务微调的基础模型，
这里仅演示接口用法，不追求分类准确率。
实际分类任务应使用在对应数据集上微调过的模型。
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


# ─────────────────────────────────────────────
# 方式一：pipeline（高层封装）
# ─────────────────────────────────────────────

print("=" * 60)
print("方式一：使用 pipeline（高层封装）")
print("=" * 60)

# pipeline 内部会自动完成：下载模型 → 初始化 tokenizer → 初始化 model → 推理 → 后处理
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I absolutely love this new feature!",
    "This is the worst experience I've ever had.",
]

results = sentiment_pipeline(texts)
print("pipeline 输出：")
for text, result in zip(texts, results):
    print(f"  文本: {text}")
    print(f"  结果: {result}")
print()


# ─────────────────────────────────────────────
# 方式二：手动 tokenize + model forward（精细控制）
# 用同一个 distilbert 模型，手动实现和 pipeline 等价的推理
# ─────────────────────────────────────────────

print("=" * 60)
print("方式二：手动 tokenize + forward（精细控制）")
print("=" * 60)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # 切换到推理模式，关闭 dropout

print(f"模型标签映射: {model.config.id2label}")
print()

# 步骤 1：Tokenization
# tokenizer 的核心工作：文本 → token id
# padding=True：批量输入时，短序列补齐到最长序列的长度
# truncation=True：超过 max_length 时截断
# max_length=512：BERT 支持的最大序列长度
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"  # 返回 PyTorch tensor
)

print("步骤 1：Tokenization 结果")
print(f"  input_ids 形状：{encoded['input_ids'].shape}  (batch_size, seq_len)")
print(f"  attention_mask 形状：{encoded['attention_mask'].shape}")

# 打印第一个样本的 token
tokens_0 = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
print(f"  第一个样本的 tokens：{tokens_0}")
print()

# 步骤 2：模型前向传播
# model(**encoded) 等价于 model(input_ids=..., attention_mask=...)
print("步骤 2：模型前向传播")
with torch.no_grad():
    outputs = model(**encoded)

# outputs.logits 形状：[batch_size, num_labels]
# 未经 softmax 的原始分数，可以是任意实数
logits = outputs.logits
print(f"  logits 形状：{logits.shape}  (batch_size, num_labels)")
print(f"  原始 logits：\n  {logits}")
print()

# 步骤 3：logits → 概率（Softmax）
print("步骤 3：Softmax → 概率")
probabilities = F.softmax(logits, dim=-1)
print(f"  概率分布：\n  {probabilities}")
print()

# 步骤 4：取概率最高的类别
print("步骤 4：argmax → 预测类别")
predicted_class_ids = torch.argmax(probabilities, dim=-1)
for i, (text, class_id) in enumerate(zip(texts, predicted_class_ids)):
    label = model.config.id2label[class_id.item()]
    score = probabilities[i][class_id].item()
    print(f"  文本: {text}")
    print(f"  预测: {label}  置信度: {score:.4f}")
print()


# ─────────────────────────────────────────────
# 对比总结：pipeline 封装了什么
# ─────────────────────────────────────────────

print("=" * 60)
print("对比总结")
print("=" * 60)
print("""
pipeline 封装了以下步骤：
  1. 根据任务名选择默认模型（或使用指定模型）
  2. 初始化 tokenizer
  3. 初始化 model
  4. tokenize 输入文本（自动处理 padding、truncation）
  5. 模型前向传播
  6. logits → softmax → argmax
  7. 把 class_id 转为标签字符串
  8. 返回格式化的结果字典

手动方式的优势：
  - 可以访问中间结果（logits、hidden_states、attention weights）
  - 可以自定义 tokenization 参数（截断策略、特殊 token 处理）
  - 可以做批量推理并手动管理 DataLoader
  - 可以在分布式环境或 GPU 上灵活控制
""")


# ─────────────────────────────────────────────
# 附加演示：获取 embedding 向量（无分类头）
# 当需要句子向量用于语义搜索、聚类时
# ─────────────────────────────────────────────

print("=" * 60)
print("附加演示：获取 embedding 向量（BertModel，无分类头）")
print("=" * 60)

from transformers import AutoModel

embed_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# AutoModel 不带任务头，输出所有 token 的隐层状态
embed_model = AutoModel.from_pretrained("bert-base-uncased")
embed_model.eval()

sentence = "Semantic search requires good sentence embeddings."
inputs = embed_tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = embed_model(**inputs)

# last_hidden_state 形状：[batch_size, seq_len, hidden_size]
last_hidden_state = outputs.last_hidden_state

# 方式 A：取 [CLS] token 的向量（位置 0）
cls_embedding = last_hidden_state[:, 0, :]  # [1, 768]

# 方式 B：对所有 token 做均值池化（通常效果更好）
mean_embedding = last_hidden_state.mean(dim=1)  # [1, 768]

print(f"句子: {sentence}")
print(f"token 数量: {inputs['input_ids'].shape[1]}")
print(f"[CLS] embedding 维度: {cls_embedding.shape}  → {cls_embedding.shape[-1]}维向量")
print(f"Mean pooling 维度: {mean_embedding.shape}  → {mean_embedding.shape[-1]}维向量")
print()
print("这些向量可以用于：")
print("  - 余弦相似度计算")
print("  - 向量数据库索引（Faiss、Milvus、pgvector）")
print("  - RAG 系统的文档检索")
