"""
第1章：Embedding 演示
使用 bert-base-uncased 提取文本 embedding，并计算余弦相似度
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# 加载 tokenizer 和模型
# BertModel 返回所有层的隐藏状态，BertForXxx 是用于特定任务的封装
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 推理时关闭梯度计算，节省内存
model.eval()

print("=" * 60)
print("1. 提取单句 Embedding")
print("=" * 60)

sentence = "The cat sat on the mat."

# 编码成 Token IDs，return_tensors="pt" 返回 PyTorch tensor
# padding 和 truncation 保证输入格式一致
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

print(f"句子: {sentence}")
print(f"input_ids shape: {inputs['input_ids'].shape}")
# shape: [batch_size=1, seq_len]

with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state 是最后一层的输出，shape: [batch_size, seq_len, hidden_size]
last_hidden_state = outputs.last_hidden_state
print(f"last_hidden_state shape: {last_hidden_state.shape}")
# 对于 bert-base-uncased，hidden_size = 768

# 取 [CLS] token 的向量作为整句的 embedding（索引 0 是 [CLS]）
cls_embedding = last_hidden_state[0, 0, :]
print(f"[CLS] embedding shape: {cls_embedding.shape}")
print(f"[CLS] embedding（前8维）: {cls_embedding[:8].numpy().round(4)}")

print()
print("=" * 60)
print("2. 计算两个句子的余弦相似度")
print("=" * 60)


def get_sentence_embedding(text: str) -> torch.Tensor:
    """
    获取句子的 [CLS] embedding，归一化后返回
    归一化使得可以直接用点积代替余弦相似度
    """
    # 编码输入
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**enc)
    # 取 [CLS] 向量，shape: [hidden_size]
    cls_vec = out.last_hidden_state[0, 0, :]
    return cls_vec


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    计算两个向量的余弦相似度
    余弦相似度 = dot(a, b) / (||a|| * ||b||)
    值域 [-1, 1]，越接近 1 越相似
    """
    # F.cosine_similarity 需要 2D 输入，unsqueeze(0) 添加 batch 维度
    sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
    return sim.item()


# 语义相近的句子对
similar_pairs = [
    ("I love cats.", "I adore kittens."),
    ("The stock market crashed today.", "Markets fell sharply this morning."),
]

# 语义不相关的句子对
dissimilar_pairs = [
    ("I love cats.", "The stock market crashed today."),
    ("Machine learning is fascinating.", "She bought three apples at the market."),
]

print("语义相近句对：")
for sent_a, sent_b in similar_pairs:
    emb_a = get_sentence_embedding(sent_a)
    emb_b = get_sentence_embedding(sent_b)
    sim = cosine_similarity(emb_a, emb_b)
    print(f"  [{sim:.4f}] '{sent_a}' ↔ '{sent_b}'")

print()
print("语义不相关句对：")
for sent_a, sent_b in dissimilar_pairs:
    emb_a = get_sentence_embedding(sent_a)
    emb_b = get_sentence_embedding(sent_b)
    sim = cosine_similarity(emb_a, emb_b)
    print(f"  [{sim:.4f}] '{sent_a}' ↔ '{sent_b}'")

print()
print("相近句对的相似度应明显高于不相关句对")

print()
print("=" * 60)
print("3. 均值池化（Mean Pooling）— 另一种句子向量获取方式")
print("=" * 60)


def mean_pooling(text: str) -> torch.Tensor:
    """
    对所有 token 的 embedding 取均值，作为句子向量
    比只用 [CLS] 更充分利用序列信息
    attention_mask 用于排除 [PAD] token 的影响
    """
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model(**enc)

    # last_hidden_state: [1, seq_len, hidden_size]
    token_embeddings = out.last_hidden_state

    # attention_mask: [1, seq_len]，pad 位置为 0，有效 token 为 1
    attention_mask = enc["attention_mask"]

    # 扩展 mask 维度以便和 embedding 相乘
    # mask_expanded: [1, seq_len, hidden_size]
    mask_expanded = attention_mask.unsqueeze(-1).float()

    # 只对有效 token 的 embedding 求和，然后除以有效 token 数量
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_vec = sum_embeddings / sum_mask

    # 去掉 batch 维度，返回 [hidden_size] 的向量
    return mean_vec.squeeze(0)


sent_a = "I love cats."
sent_b = "I adore kittens."
sent_c = "The stock market crashed today."

emb_a = mean_pooling(sent_a)
emb_b = mean_pooling(sent_b)
emb_c = mean_pooling(sent_c)

print(f"Mean pooling embedding shape: {emb_a.shape}")
print(f"相似句对相似度: {cosine_similarity(emb_a, emb_b):.4f}  ('{sent_a}' ↔ '{sent_b}')")
print(f"不相关句对相似度: {cosine_similarity(emb_a, emb_c):.4f}  ('{sent_a}' ↔ '{sent_c}')")
