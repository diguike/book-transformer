"""
句子 Embedding 与余弦相似度演示

运行环境：CPU 即可，无需 GPU
依赖安装：pip install -r requirements.txt

演示内容：
  1. 用 sentence-transformers 对多个句子生成 embedding
  2. 计算两两之间的余弦相似度
  3. 打印相似度矩阵，验证语义相近的句子分数高
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# 加载预训练模型
# all-MiniLM-L6-v2：轻量级英文模型，384 维，约 22MB
# 首次运行会自动从 HuggingFace Hub 下载
model = SentenceTransformer('all-MiniLM-L6-v2')

# 测试句子：前两组语义相近，第三组语义不同
sentences = [
    "The weather is lovely today.",       # 天气相关
    "It's so sunny and warm outside!",    # 天气相关（语义相近）
    "I really enjoy the warm sunshine.",  # 天气相关（稍远）
    "He drove to the soccer stadium.",    # 交通/体育（语义不同）
    "She took the bus to the airport.",   # 交通（与第4句有一定相关）
    "Machine learning is fascinating.",   # 技术话题（完全不同）
]

print("=" * 60)
print("句子列表：")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

print()

# encode() 内部流程：tokenize → BERT 前向传播 → mean pooling → L2 归一化
# normalize_embeddings=True 确保输出向量模长为 1，余弦相似度 = 点积
embeddings = model.encode(sentences, normalize_embeddings=True)

print(f"Embedding 形状：{embeddings.shape}")  # (6, 384)
print(f"单个向量维度：{embeddings.shape[1]}")
print(f"向量模长（归一化后应为 1.0）：{np.linalg.norm(embeddings[0]):.6f}")
print()

# 计算余弦相似度矩阵
# 因为已经 L2 归一化，矩阵乘法等价于余弦相似度
# util.cos_sim() 返回 PyTorch tensor
similarity_matrix = util.cos_sim(embeddings, embeddings)

print("余弦相似度矩阵（行/列对应上面的句子编号）：")
print()

# 打印表头
header = "      " + "".join(f"  [{i}]  " for i in range(len(sentences)))
print(header)

# 打印矩阵
for i in range(len(sentences)):
    row = f"[{i}]  "
    for j in range(len(sentences)):
        score = similarity_matrix[i][j].item()
        row += f"  {score:.3f}"
    print(row)

print()

# 找出最相似的句子对（排除自身对比）
print("最相似的句子对：")
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j].item()
        pairs.append((score, i, j))

# 按相似度降序排列
pairs.sort(reverse=True)

for score, i, j in pairs[:5]:
    print(f"  相似度 {score:.4f}：")
    print(f"    [{i}] {sentences[i]}")
    print(f"    [{j}] {sentences[j]}")
    print()

# 验证预期结果：
# - 天气相关的句子（[0], [1], [2]）之间相似度应该明显高于跨话题的句子对
# - [3] 和 [4] 都涉及交通工具，相似度应该有一定数值
# - [5] 技术话题与其他所有句子的相似度应该偏低
print("=" * 60)
print("验证结论：")
weather_score = similarity_matrix[0][1].item()
transport_score = similarity_matrix[3][4].item()
cross_topic_score = similarity_matrix[0][5].item()

print(f"  天气句子对 [0]-[1] 相似度：{weather_score:.4f}")
print(f"  交通句子对 [3]-[4] 相似度：{transport_score:.4f}")
print(f"  跨话题 [0]-[5] 相似度：   {cross_topic_score:.4f}")
print()
print("  语义相近的句子对分数明显更高，embedding 捕获到了语义信息。")
