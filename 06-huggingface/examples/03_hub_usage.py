"""
HuggingFace Hub 使用演示

演示内容：
1. 从 Hub 加载模型（from_pretrained）
2. 查看模型配置信息（config）
3. 本地缓存路径

运行前安装依赖：
    pip install -r requirements.txt
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


# ─────────────────────────────────────────────
# 1. 从 Hub 加载模型
#    from_pretrained 的工作流程：
#    1) 检查本地缓存是否存在
#    2) 不存在则从 Hub 下载（需要网络）
#    3) 缓存到本地，下次直接加载
# ─────────────────────────────────────────────

print("=" * 60)
print("1. 从 Hub 加载模型")
print("=" * 60)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"加载模型: {model_name}")
print("首次运行会从 Hub 下载，后续从本地缓存加载...")
print()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(f"模型加载成功")
print(f"模型类型: {type(model).__name__}")
print(f"Tokenizer 类型: {type(tokenizer).__name__}")
print()


# ─────────────────────────────────────────────
# 2. 查看模型配置信息（config）
#    config 包含了模型的所有超参数和元信息
#    这是判断模型是否适合你任务的第一手资料
# ─────────────────────────────────────────────

print("=" * 60)
print("2. 查看模型配置（config）")
print("=" * 60)

config = model.config

# 基础架构信息
print("【基础架构】")
print(f"  模型类型 (model_type): {config.model_type}")
print(f"  隐层维度 (hidden_size): {config.hidden_size}")
print(f"  Transformer 层数 (num_hidden_layers): {config.num_hidden_layers}")
print(f"  注意力头数 (num_attention_heads): {config.num_attention_heads}")
print(f"  FFN 中间层维度 (intermediate_size): {config.intermediate_size}")
print(f"  最大序列长度 (max_position_embeddings): {config.max_position_embeddings}")
print()

# 任务相关信息
print("【任务配置】")
print(f"  标签数量 (num_labels): {config.num_labels}")
print(f"  标签映射 id2label: {config.id2label}")
print(f"  标签映射 label2id: {config.label2id}")
print()

# 词表信息
print("【词表信息】")
print(f"  词表大小 (vocab_size): {config.vocab_size}")
print()

# 模型来源信息（从模型卡片继承）
print("【模型元信息】")
if hasattr(config, '_name_or_path'):
    print(f"  模型名称: {config._name_or_path}")
print()


# ─────────────────────────────────────────────
# 3. 不加载权重，只查看 config
#    适合快速了解模型规格而不消耗内存
# ─────────────────────────────────────────────

print("=" * 60)
print("3. 仅查看配置（不加载权重）")
print("=" * 60)

# AutoConfig.from_pretrained 只下载/读取 config.json，不加载模型权重
# 适合：快速了解模型规格、对比多个模型时节省内存
config_only = AutoConfig.from_pretrained("bert-base-uncased")
print(f"bert-base-uncased 配置（不加载权重）:")
print(f"  架构: {config_only.model_type}, {config_only.num_hidden_layers} 层, {config_only.hidden_size} 维")
print(f"  注意力头: {config_only.num_attention_heads} 个")
print(f"  词表大小: {config_only.vocab_size}")
print()


# ─────────────────────────────────────────────
# 4. 本地缓存路径
#    了解缓存位置，便于管理磁盘空间
# ─────────────────────────────────────────────

print("=" * 60)
print("4. 本地缓存路径")
print("=" * 60)

# 默认缓存目录由以下环境变量控制（优先级从高到低）：
# HF_HOME > HUGGINGFACE_HUB_CACHE > ~/.cache/huggingface
hf_home = os.environ.get("HF_HOME", "")
hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "")
default_cache = os.path.expanduser("~/.cache/huggingface/hub")

print("缓存目录查找顺序：")
print(f"  1. HF_HOME 环境变量: {'已设置 → ' + hf_home if hf_home else '未设置'}")
print(f"  2. HUGGINGFACE_HUB_CACHE 环境变量: {'已设置 → ' + hf_cache if hf_cache else '未设置'}")
print(f"  3. 默认路径: {default_cache}")
print()

# 检查默认缓存目录下有哪些已下载的模型
if os.path.exists(default_cache):
    cached_items = [d for d in os.listdir(default_cache) if d.startswith("models--")]
    if cached_items:
        print(f"已缓存的模型（在 {default_cache}）：")
        for item in sorted(cached_items):
            # 目录名格式：models--<org>--<name> → 转换为 <org>/<name>
            model_id = item.replace("models--", "").replace("--", "/")
            model_cache_path = os.path.join(default_cache, item)
            # 计算目录大小
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_cache_path):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    if os.path.isfile(fpath):
                        total_size += os.path.getsize(fpath)
            size_mb = total_size / (1024 * 1024)
            print(f"  {model_id:<55} {size_mb:.1f} MB")
    else:
        print("缓存目录存在但尚无缓存模型")
else:
    print(f"默认缓存目录不存在: {default_cache}")
print()

print("=" * 60)
print("常用环境变量配置示例")
print("=" * 60)
print("""
# 修改缓存路径（写入 ~/.bashrc 或 ~/.zshrc）
export HF_HOME=/data/hf_cache

# 离线模式：只从本地缓存加载，不发送网络请求
export TRANSFORMERS_OFFLINE=1

# 国内镜像加速（hf-mirror.com 是常用的国内镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 在 Python 代码中也可以动态设置
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
""")
