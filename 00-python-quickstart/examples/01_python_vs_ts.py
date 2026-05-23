"""
Python 与 TypeScript 语法对比示例
演示两种语言在变量、函数、类、列表推导式、字典操作上的对应关系
"""

# ============================================================
# 1. 变量声明
# TS 中等价写法是: const name: string = "transformer"
# ============================================================

name = "transformer"
count = 512
ratio = 0.1
is_ready = True

print("=== 变量 ===")
print(f"name={name}, count={count}, ratio={ratio}, is_ready={is_ready}")
# f-string 是 Python 的字符串插值，对应 TS 的模板字符串 `name=${name}`


# ============================================================
# 2. 列表操作
# TS 中等价写法是: const tokens: string[] = ["hello", "world"]
# ============================================================

print("\n=== 列表 ===")
tokens = ["hello", "world", "transformer"]

# 追加元素 — TS: tokens.push("new")
tokens.append("new")

# 索引访问 — TS: tokens[0]
print("第一个元素:", tokens[0])

# 负索引 — TS 没有直接等价，需要 tokens[tokens.length - 1]
print("最后一个元素:", tokens[-1])

# 切片 — TS: tokens.slice(1, 3)
print("切片 [1:3]:", tokens[1:3])

# 长度 — TS: tokens.length
print("长度:", len(tokens))


# ============================================================
# 3. 列表推导式
# TS 中等价写法是 Array.map() 和 Array.filter()
# ============================================================

print("\n=== 列表推导式 ===")

# map: 把每个元素转大写
# TS: tokens.map(t => t.toUpperCase())
uppercased = [t.upper() for t in tokens]
print("大写:", uppercased)

# filter: 只保留长度大于 4 的 token
# TS: tokens.filter(t => t.length > 4)
long_tokens = [t for t in tokens if len(t) > 4]
print("长度>4:", long_tokens)

# map + filter 组合
# TS: tokens.filter(t => t.length > 4).map(t => t.upper())
result = [t.upper() for t in tokens if len(t) > 4]
print("先过滤再大写:", result)


# ============================================================
# 4. 字典操作
# TS 中等价写法是: const config: Record<string, any> = { ... }
# ============================================================

print("\n=== 字典 ===")
config = {
    "model": "bert-base-uncased",
    "max_length": 512,
    "do_lower_case": True,
    "vocab_size": 30522,
}

# 访问键 — TS: config.model 或 config["model"]
print("model:", config["model"])

# 带默认值的访问 — TS: config.missing ?? "default"
print("missing key:", config.get("missing_key", "default_value"))

# 添加/修改键 — TS: config.newKey = "value"
config["batch_size"] = 32
print("添加后:", config)

# 遍历键值对 — TS: Object.entries(config).forEach(([k, v]) => ...)
print("\n遍历所有配置项:")
for key, value in config.items():
    print(f"  {key}: {value}")

# 检查键是否存在 — TS: "model" in config
print("\n'model' 存在:", "model" in config)
print("'learning_rate' 存在:", "learning_rate" in config)


# ============================================================
# 5. 函数定义
# TS 中等价写法是:
# function tokenize(text: string, maxLength: number = 512): string[]
# ============================================================

print("\n=== 函数 ===")

def tokenize(text, max_length=512, lowercase=True):
    """
    将文本分割为 token 列表

    参数:
        text: 输入文本
        max_length: 最大 token 数量
        lowercase: 是否转小写

    返回:
        token 列表
    """
    if lowercase:
        text = text.lower()
    # 简单按空格切分（实际使用时会用专业 tokenizer）
    tokens = text.split()
    # 截断到 max_length — 对应 TS: tokens.slice(0, maxLength)
    return tokens[:max_length]


# 普通调用
result = tokenize("Hello World Transformer")
print("分词结果:", result)

# 关键字参数调用 — TS 不支持关键字参数，只能按顺序
result = tokenize("Hello World", max_length=1, lowercase=False)
print("限制长度+不转小写:", result)


# ============================================================
# 6. 类定义
# TS 中等价写法是 class TokenizerConfig { ... }
# ============================================================

print("\n=== 类 ===")

class TokenizerConfig:
    # 类属性 — TS: static defaultPadToken = "[PAD]"
    default_pad_token = "[PAD]"

    def __init__(self, vocab_size, max_length=512):
        # __init__ 是构造函数，self 是实例引用，对应 TS 的 this
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token = TokenizerConfig.default_pad_token

    def summary(self):
        # 实例方法必须有 self 参数 — TS 不需要显式声明 this
        return f"TokenizerConfig(vocab={self.vocab_size}, max_len={self.max_length})"

    def is_large_vocab(self):
        # 超过 5 万词汇表认为是"大词汇"
        return self.vocab_size > 50000

    @staticmethod
    def default():
        # 静态方法，不依赖实例 — TS: static default(): TokenizerConfig
        return TokenizerConfig(vocab_size=30522)

    @classmethod
    def from_dict(cls, d):
        # 类方法，第一个参数是类本身 — 常用于工厂方法
        # TS 中通常用 static 工厂方法实现
        return cls(
            vocab_size=d.get("vocab_size", 30522),
            max_length=d.get("max_length", 512),
        )


# 实例化 — TS: new TokenizerConfig(30522, 256)
config = TokenizerConfig(vocab_size=30522, max_length=256)
print(config.summary())
print("大词汇表:", config.is_large_vocab())

# 静态方法调用
default_config = TokenizerConfig.default()
print("默认配置:", default_config.summary())

# 类方法（工厂方法）调用
config_from_dict = TokenizerConfig.from_dict({"vocab_size": 50000, "max_length": 1024})
print("从字典创建:", config_from_dict.summary())


# ============================================================
# 7. 元组解构
# TS 中等价写法是: const [batchSize, dim] = shape
# ============================================================

print("\n=== 元组解构 ===")
shape = (32, 10, 512)       # batch_size, seq_len, hidden_dim
batch_size, seq_len, hidden_dim = shape
print(f"batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")


print("\n所有示例运行完毕。")
