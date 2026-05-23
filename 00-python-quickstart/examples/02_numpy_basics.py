"""
NumPy 核心操作示例
覆盖后续章节用到的所有基础操作：
创建数组/矩阵、元素级运算、矩阵乘法、shape/reshape、切片
"""

import numpy as np

print("NumPy 版本:", np.__version__)


# ============================================================
# 1. 创建数组和矩阵
# ============================================================

print("\n=== 创建数组 ===")

# 一维数组（向量）
v = np.array([1.0, 2.0, 3.0, 4.0])
print("一维数组:", v)
print("shape:", v.shape)     # (4,) — 4 个元素，一维
print("dtype:", v.dtype)     # float64

# 二维数组（矩阵）
# 想象成 3 个 token，每个 token 用 4 维向量表示（Embedding 场景）
embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4],   # token 0 的向量表示
    [0.5, 0.6, 0.7, 0.8],   # token 1 的向量表示
    [0.9, 1.0, 1.1, 1.2],   # token 2 的向量表示
])
print("\n二维数组（Embedding 矩阵）:")
print(embeddings)
print("shape:", embeddings.shape)    # (3, 4): 3 个 token，4 维

# 常用初始化方法
print("\n全零矩阵 (2x3):")
print(np.zeros((2, 3)))

print("\n全一矩阵 (2x3):")
print(np.ones((2, 3)))

print("\n单位矩阵 (4x4):")
print(np.eye(4))

# 随机初始化（标准正态分布）— 模型权重的典型初始化方式
np.random.seed(42)          # 固定随机种子，保证结果可复现
weights = np.random.randn(4, 3)
print("\n随机正态分布矩阵 (4x3):")
print(weights)

# 等差数列
print("\nnp.arange(0, 10, 2):", np.arange(0, 10, 2))    # [0 2 4 6 8]
print("np.linspace(0, 1, 5):", np.linspace(0, 1, 5))   # 5 个均匀分布的点


# ============================================================
# 2. 元素级运算（不需要写循环）
# ============================================================

print("\n=== 元素级运算 ===")

a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([10.0, 20.0, 30.0, 40.0])

print("a:", a)
print("b:", b)
print("a + b:", a + b)           # 对应元素相加
print("a - b:", a - b)           # 对应元素相减
print("a * b:", a * b)           # 对应元素相乘（不是矩阵乘法！）
print("a / b:", a / b)           # 对应元素相除
print("a ** 2:", a ** 2)         # 每个元素的平方
print("np.sqrt(a):", np.sqrt(a)) # 每个元素开方
print("np.exp(a):", np.exp(a))   # e^x，在 softmax 里会用到

# 标量广播 — 标量会自动扩展到数组的形状
print("\n标量广播:")
print("a * 0.5:", a * 0.5)       # 每个元素乘以 0.5
print("a + 100:", a + 100)       # 每个元素加 100

# 广播（Broadcasting）— NumPy 的核心特性
# 形状不同的数组在满足条件时可以直接运算
print("\n广播示例:")
matrix = np.ones((3, 4))
row_bias = np.array([1.0, 2.0, 3.0, 4.0])   # shape: (4,)
# matrix shape (3,4) 和 row_bias shape (4,) 可以广播
result = matrix + row_bias
print("matrix + row_bias (广播):\n", result)


# ============================================================
# 3. 矩阵乘法
# Attention 机制的核心计算依赖矩阵乘法
# ============================================================

print("\n=== 矩阵乘法 ===")

# 向量点积：两个向量对应元素相乘后求和
# 用于计算两个 Embedding 的相似度
query = np.array([1.0, 0.5, -0.5, 0.2])
key   = np.array([0.8, 0.3, -0.4, 0.1])
dot_product = np.dot(query, key)
print("向量点积 (相似度):", dot_product)

# 矩阵乘法：@ 运算符
# (3, 4) @ (4, 2) => (3, 2)
# 3 个 token 的 4 维向量，乘以 4x2 的权重矩阵，得到 3 个 2 维向量
np.random.seed(42)
token_embeddings = np.random.randn(3, 4)    # 3 个 token，每个 4 维
weight_matrix    = np.random.randn(4, 2)    # 投影权重
output = token_embeddings @ weight_matrix   # 矩阵乘法
print("\n矩阵乘法 (3,4) @ (4,2):")
print("输入 shape:", token_embeddings.shape)
print("权重 shape:", weight_matrix.shape)
print("输出 shape:", output.shape)           # (3, 2)

# np.matmul 和 @ 等价
output2 = np.matmul(token_embeddings, weight_matrix)
print("两种写法结果相同:", np.allclose(output, output2))

# 批量矩阵乘法：第一维是 batch
# (batch, seq, dim) @ (batch, dim, out) => (batch, seq, out)
batch_size, seq_len, dim = 2, 3, 4
batch_tokens  = np.random.randn(batch_size, seq_len, dim)
batch_weights = np.random.randn(batch_size, dim, 2)
batch_output  = batch_tokens @ batch_weights
print("\n批量矩阵乘法:", batch_tokens.shape, "@", batch_weights.shape, "=>", batch_output.shape)


# ============================================================
# 4. shape 与 reshape
# ============================================================

print("\n=== Shape 与 Reshape ===")

# 典型的 NLP 数据维度：(batch_size, seq_len, hidden_dim)
batch = np.zeros((32, 10, 512))
print("典型 batch shape:", batch.shape)     # (32, 10, 512)
print("总元素数:", batch.size)               # 32 * 10 * 512 = 163840

# reshape：改变形状，总元素数不变
flat = batch.reshape(32, -1)                # -1 让 NumPy 自动推断
print("reshape 后:", flat.shape)             # (32, 5120)

# 展平成一维
flat_all = batch.reshape(-1)
print("完全展平:", flat_all.shape)           # (163840,)

# 转置
matrix = np.ones((3, 4))
print("\n原矩阵 shape:", matrix.shape)       # (3, 4)
print("转置后 shape:", matrix.T.shape)       # (4, 3)

# 对多维数组转置特定维度（Attention 里常用）
# (batch, heads, seq, dim) 转置 seq 和 dim
tensor = np.zeros((2, 8, 10, 64))
transposed = tensor.transpose(0, 1, 3, 2)   # 交换最后两个维度
print("\n转置前:", tensor.shape)              # (2, 8, 10, 64)
print("转置后:", transposed.shape)           # (2, 8, 64, 10)

# expand_dims：增加维度（广播前经常需要）
v = np.array([1.0, 2.0, 3.0])               # shape: (3,)
v_row = v[np.newaxis, :]                     # shape: (1, 3)，添加行维度
v_col = v[:, np.newaxis]                     # shape: (3, 1)，添加列维度
print("\n原向量 shape:", v.shape)
print("行向量 shape:", v_row.shape)
print("列向量 shape:", v_col.shape)


# ============================================================
# 5. 切片
# ============================================================

print("\n=== 切片 ===")

# 构造一个 3x4 的矩阵，值是 0-11
data = np.arange(12).reshape(3, 4)
print("原始矩阵:")
print(data)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print("\n第一行 data[0]:", data[0])           # [0 1 2 3]
print("第一列 data[:,0]:", data[:, 0])        # [0 4 8]
print("子矩阵 data[0:2,1:3]:\n", data[0:2, 1:3])   # [[1 2],[5 6]]
print("每隔一行 data[::2]:\n", data[::2])    # [[0 1 2 3],[8 9 10 11]]
print("最后一行 data[-1]:", data[-1])         # [8 9 10 11]
print("最后一列 data[:,-1]:", data[:, -1])    # [3 7 11]

# 布尔索引 — 根据条件筛选元素
v = np.array([1, -2, 3, -4, 5])
mask = v > 0                                  # [True False True False True]
print("\n布尔索引，只取正数:", v[mask])        # [1 3 5]


# ============================================================
# 6. 常用统计函数
# ============================================================

print("\n=== 统计函数 ===")

scores = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print("全局最大值:", scores.max())              # 6.0
print("按行最大值:", scores.max(axis=1))        # [3. 6.]
print("按列最大值:", scores.max(axis=0))        # [4. 5. 6.]
print("全局均值:", scores.mean())               # 3.5
print("按行求和:", scores.sum(axis=1))          # [6. 15.]

# softmax 手动实现（Attention 里会用到）
# softmax(x_i) = exp(x_i) / sum(exp(x))
def softmax(x):
    # 减去最大值避免数值溢出（exp(大数) 会爆）
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print("\nsoftmax([2.0, 1.0, 0.1]):", probs)
print("概率之和:", probs.sum())                  # 应该等于 1.0


print("\n所有示例运行完毕。")
