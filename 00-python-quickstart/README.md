---
title: 第0章：给工程师的 Python 速成
feishu_url: "https://www.feishu.cn/wiki/Sv4DwsiPYieLxEkN4hScqwPwnlg"
last_synced: "2026-05-07T23:06:16"
---

## 为什么是 Python

不是因为 Python 更好，而是因为 AI 生态完全建立在 Python 上。

PyTorch、TensorFlow、HuggingFace Transformers、NumPy、scikit-learn——这些库没有成熟的 JS 替代品。不是没有人尝试过，而是生态差距太大，基本没有可比性。想用这些工具，就得会 Python。

这不意味着你要放弃 TypeScript。第 12 章会讲怎么在 TypeScript 项目里集成模型能力。但理解和调试模型，Python 是唯一现实的选择。

好消息是：如果你会 TypeScript，Python 的学习成本很低。Python 的变量没有类型声明，类型由赋值时的值决定；TypeScript 是静态类型语言，编译器会检查类型，类型注解在很多场景下可以由编译器自动推断，但类型约束始终存在。两者语法结构相似，标准库都很完善。主要的差异是缩进语法、没有花括号、以及一些内置数据结构的习惯用法。

本章只覆盖后续章节用到的 Python 特性，不是完整的语言教程。

## 类型与变量

Python 的变量不需要声明类型，赋值即声明：

```python
# Python
name = "transformer"
count = 512
ratio = 0.1
is_ready = True
```

```typescript
// TypeScript 等价写法
const name = "transformer";
const count = 512;
const ratio = 0.1;
const isReady = true;
```

Python 有四种核心容器类型：

```python
# 列表（有序，可变）— 对应 JS Array
tokens = ["hello", "world", "!"]
tokens.append("new")          # 末尾追加
tokens[0]                     # 索引访问，结果是 "hello"
tokens[-1]                    # 负索引，结果是 "new"（最后一个）
tokens[1:3]                   # 切片，结果是 ["world", "!"]

# 元组（有序，不可变）— 对应 JS readonly Array
shape = (32, 512)             # 批大小和维度
batch_size, dim = shape       # 解构赋值

# 字典（键值对）— 对应 JS 对象/Map
config = {
    "model": "bert-base",
    "max_length": 512,
    "do_lower_case": True,
}
config["model"]               # 访问键，结果是 "bert-base"
config.get("missing", "default")  # 带默认值的访问

# 集合（无序，唯一）— 对应 JS Set
vocab = {"apple", "banana", "cherry"}
"apple" in vocab              # 成员检查，结果是 True
```

**列表推导式**是 Python 最常用的语法糖，相当于 JS 的 `map` + `filter`：

```python
# JS: tokens.map(t => t.lower())
lowercased = [t.lower() for t in tokens]

# JS: tokens.filter(t => t.length > 3)
long_tokens = [t for t in tokens if len(t) > 3]

# JS: tokens.filter(t => t.length > 3).map(t => t.upper())
result = [t.upper() for t in tokens if len(t) > 3]
```

## 函数与类

函数用 `def` 定义，参数可以有默认值，也支持关键字参数调用：

```python
# Python 函数定义
# 注意：这里的 tokenize 只是演示 Python 函数写法，按空格切分不是真正的 tokenization。
# 第 1 章会介绍 BPE 等真实的分词方法。
def tokenize(text, max_length=512, lowercase=True):
    """将文本转换为 token 列表"""
    if lowercase:
        text = text.lower()
    tokens = text.split()
    return tokens[:max_length]

# 调用方式
tokenize("Hello World")                          # 用默认参数
tokenize("Hello World", max_length=10)           # 关键字参数
tokenize("Hello World", max_length=10, lowercase=False)
```

```typescript
// TypeScript 等价写法
function tokenize(
  text: string,
  maxLength: number = 512,
  lowercase: boolean = true
): string[] {
  if (lowercase) text = text.toLowerCase();
  const tokens = text.split(" ");
  return tokens.slice(0, maxLength);
}
```

Python 的类语法和 TypeScript 相似，但用 `self` 代替 `this`，且必须显式传入：

```python
# Python 类定义
class TokenizerConfig:
    def __init__(self, vocab_size, max_length=512):
        # __init__ 是构造函数，self 相当于 TS 的 this
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token = "[PAD]"

    def summary(self):
        return f"vocab={self.vocab_size}, max_len={self.max_length}"

    @staticmethod
    def default():
        # 静态方法，不需要实例
        return TokenizerConfig(vocab_size=30522)


# 使用
config = TokenizerConfig(vocab_size=30522, max_length=256)
print(config.summary())       # "vocab=30522, max_len=256"
```

```typescript
// TypeScript 等价写法
class TokenizerConfig {
  vocabSize: number;
  maxLength: number;
  padToken: string = "[PAD]";

  constructor(vocabSize: number, maxLength: number = 512) {
    this.vocabSize = vocabSize;
    this.maxLength = maxLength;
  }

  summary(): string {
    return `vocab=${this.vocabSize}, max_len=${this.maxLength}`;
  }

  static default(): TokenizerConfig {
    return new TokenizerConfig(30522);
  }
}
```

Python 用缩进表示代码块，没有花括号。缩进必须一致，通常是 4 个空格。配置你的编辑器统一使用空格（推荐 4 个），不要混用 Tab 和空格，否则运行时会报 `IndentationError`。

## NumPy：工程师视角的矩阵运算

NumPy 是 Python 科学计算的基础库，后续的 PyTorch 和模型相关操作都建立在它的思想上。

核心概念：`ndarray`（N 维数组）。可以把它理解成 JS Array 的超集——支持多维度，支持批量运算，底层用 C 实现所以很快。

```python
import numpy as np

# 创建一维数组（向量）
v = np.array([1.0, 2.0, 3.0])
print(v.shape)     # (3,)  — 3 个元素的一维数组
print(v.dtype)     # float64

# 创建二维数组（矩阵）
# 3 行 4 列的矩阵，代表 3 个 token，每个 token 是 4 维向量
embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4],   # token 0 的向量
    [0.5, 0.6, 0.7, 0.8],   # token 1 的向量
    [0.9, 1.0, 1.1, 1.2],   # token 2 的向量
])
print(embeddings.shape)    # (3, 4)
```

NumPy 的关键优势是**元素级运算**，不需要写循环：

```python
# JS 里你必须写循环或用 map
# Python/NumPy 里可以直接对整个数组操作

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

a + b          # array([5., 7., 9.])     — 元素级加法
a * 2          # array([2., 4., 6.])     — 标量乘法
a ** 2         # array([1., 4., 9.])     — 元素级平方
np.sqrt(a)     # array([1., 1.41, 1.73]) — 元素级开方
```

**矩阵乘法**在 Attention 计算里无处不在：

```python
# 矩阵乘法：@ 运算符或 np.matmul
# (3, 4) @ (4, 2) => (3, 2)
W = np.random.randn(4, 2)           # 权重矩阵，4 行 2 列
output = embeddings @ W             # 矩阵乘法
print(output.shape)                 # (3, 2)

# 等价写法
output = np.matmul(embeddings, W)
```

**shape 和 reshape** 在处理 batch 数据时很重要：

```python
# 一个 batch 的数据：32 个样本，每个样本 10 个 token，每个 token 512 维
batch = np.zeros((32, 10, 512))
print(batch.shape)                  # (32, 10, 512)

# reshape：改变形状，不改变数据
flat = batch.reshape(32, -1)        # -1 表示自动推断，(32, 5120)
print(flat.shape)

# 转置
matrix = np.ones((3, 4))
print(matrix.T.shape)               # (4, 3)
```

**切片**比 JS 更强大：

```python
data = np.arange(12).reshape(3, 4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

data[0]          # 第一行: [0, 1, 2, 3]
data[:, 0]       # 第一列: [0, 4, 8]
data[0:2, 1:3]   # 子矩阵: [[1,2],[5,6]]
data[::2]        # 每隔一行: [[0,1,2,3],[8,9,10,11]]
```

常用的初始化函数：

```python
np.zeros((3, 4))          # 全零矩阵
np.ones((3, 4))           # 全一矩阵
np.eye(4)                 # 4x4 单位矩阵
np.random.randn(3, 4)     # 标准正态分布随机值
np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]，类似 JS 的 range
```

这些操作在后面的 Attention 计算章节会频繁出现，到时候看到不会陌生。

## 环境配置

**推荐用 conda 管理环境。** conda 能同时管理 Python 版本和包，避免不同项目之间的依赖冲突。

安装 Miniconda（比完整 Anaconda 小很多）：

```bash
# macOS（方式一：Homebrew）
brew install --cask miniconda
# 方式二：从官网下载安装脚本（更稳定）
# 访问 https://docs.conda.io/en/latest/miniconda.html 下载对应版本

# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

创建和使用环境：

```bash
# 创建 Python 3.11 的环境
conda create -n transformer-book python=3.11

# 激活环境
conda activate transformer-book

# 安装依赖
pip install -r requirements.txt

# 查看已安装的包
pip list

# 退出环境
conda deactivate
```

如果不想用 conda，Python 内置的 venv 也够用：

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**国内网络访问 HuggingFace**

后续章节的代码会从 HuggingFace Hub 下载模型（`from_pretrained()`）。国内网络直连 HuggingFace 通常很慢或无法访问，建议配置镜像：

```bash
# 在 ~/.bashrc 或 ~/.zshrc 里加入（或每次运行前设置）
export HF_ENDPOINT=https://hf-mirror.com
```

设置后，所有 `from_pretrained()` 调用会自动走镜像地址。

**Jupyter Notebook** 是 AI 领域调试代码的标配。可以把代码分成小块（cell）逐个运行，方便查看中间结果：

```bash
# 安装
pip install jupyter

# 启动（在项目目录下）
jupyter notebook

# 或者用更现代的 JupyterLab
pip install jupyterlab
jupyter lab
```

启动后浏览器会自动打开，新建 `.ipynb` 文件即可。每个 cell 用 `Shift+Enter` 运行，输出紧跟在 cell 下方。

VS Code 也内置了 Jupyter 支持，安装 Python 扩展后可以直接在编辑器里运行 `.ipynb` 文件，不需要开浏览器。

本章的 `examples/` 目录下有两个可运行的脚本，建议跑一遍确认环境正常：

```bash
cd examples
pip install -r requirements.txt
python 01_python_vs_ts.py
python 02_numpy_basics.py
```

---

> 本章来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
