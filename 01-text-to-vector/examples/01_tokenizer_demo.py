"""
第1章：Tokenizer 演示
使用 BERT tokenizer 演示文本切分、Token ID 映射和特殊 token
"""

from transformers import BertTokenizer

# 加载 bert-base-uncased 的 tokenizer
# uncased 表示不区分大小写，所有文本会先转成小写
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("=" * 60)
print("1. 基本 tokenize 演示")
print("=" * 60)

text = "Hello, how are you doing today?"

# tokenize 只切分，不加特殊 token
tokens = tokenizer.tokenize(text)
print(f"原始文本: {text}")
print(f"切分结果: {tokens}")
# bert-base-uncased 会把 "doing" 切成 "doing"，保留完整词
# 但像 "tokenization" 这样的词会被切成子词

print()

# encode 会加上特殊 token，返回 token ids
# add_special_tokens=True 是默认值，[CLS] 在开头，[SEP] 在结尾
token_ids = tokenizer.encode(text, add_special_tokens=True)
print(f"Token IDs（含特殊token）: {token_ids}")
print(f"Token 数量: {len(token_ids)}")

print()

# convert_ids_to_tokens 可以把 id 反查回 token 字符串，方便对照
tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
print("ID → Token 对照：")
for token_id, token in zip(token_ids, tokens_with_special):
    print(f"  {token_id:5d} → {token}")

print()
print("=" * 60)
print("2. 特殊 Token 说明")
print("=" * 60)

# 查看特殊 token 对应的 ID
print(f"[CLS] token id: {tokenizer.cls_token_id}")   # 101
print(f"[SEP] token id: {tokenizer.sep_token_id}")   # 102
print(f"[PAD] token id: {tokenizer.pad_token_id}")   # 0
print(f"[MASK] token id: {tokenizer.mask_token_id}") # 103

print()
print("观察上面的输出：token_ids 第一个是 101（[CLS]），最后一个是 102（[SEP]）")

print()
print("=" * 60)
print("3. 不同文本长度对应不同 token 数量")
print("=" * 60)

# 演示不同长度的文本 token 数量差异
texts = [
    "Hi.",
    "The weather is nice today.",
    "Tokenization is the process of breaking text into smaller units called tokens.",
    "In natural language processing, tokenization refers to the task of chopping text up "
    "into pieces called tokens, which may be words, characters, or subwords.",
]

for t in texts:
    ids = tokenizer.encode(t, add_special_tokens=True)
    print(f"  [{len(ids):3d} tokens] {t[:60]}{'...' if len(t) > 60 else ''}")

print()
print("=" * 60)
print("4. 子词切分演示（BPE 效果）")
print("=" * 60)

# 演示 BPE 子词切分：不常见的词会被切成子词
special_words = [
    "tokenization",    # 会被切成 token + ##ization
    "unfortunately",   # un + ##fortunate + ##ly
    "antidisestablishmentarianism",  # 极长词，切分明显
    "GPT",             # 缩写
    "transformer",     # 常见词，不一定切分
]

print("词 → 子词切分结果：")
for word in special_words:
    sub_tokens = tokenizer.tokenize(word)
    print(f"  {word:35s} → {sub_tokens}")

print()
print("注意：## 前缀表示这是一个词的非首部子词（continuation subword）")

print()
print("=" * 60)
print("5. 词表大小")
print("=" * 60)
print(f"bert-base-uncased 词表大小: {tokenizer.vocab_size}")
