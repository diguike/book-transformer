"""
Pipeline 快速上手

演示 5 种常用 pipeline：
1. sentiment-analysis  — 情感分析
2. ner                 — 命名实体识别
3. question-answering  — 抽取式问答
4. summarization       — 文本摘要
5. text-generation     — 文本生成

运行前安装依赖：
    pip install -r requirements.txt

首次运行会下载模型，需要网络连接（或配置 HF_ENDPOINT 镜像）。
"""

from transformers import pipeline


# ─────────────────────────────────────────────
# 1. 情感分析
#    典型应用：用户评价分析、舆情监控、内容审核
#    输出：正面/负面标签 + 置信度
# ─────────────────────────────────────────────

print("=" * 60)
print("1. 情感分析（Sentiment Analysis）")
print("=" * 60)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "This product exceeded all my expectations!",
    "The customer service was absolutely terrible.",
    "It's an okay product, nothing special.",
]

results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"文本: {text}")
    print(f"预测: {result['label']}  置信度: {result['score']:.4f}")
    print()


# ─────────────────────────────────────────────
# 2. 命名实体识别（NER）
#    典型应用：信息抽取、知识图谱构建、合同要素提取
#    输出：实体文本、实体类型（PER/ORG/LOC 等）、位置
# ─────────────────────────────────────────────

print("=" * 60)
print("2. 命名实体识别（NER）")
print("=" * 60)

ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    # aggregation_strategy="simple" 会把同一实体的多个 token 合并
    # 比如 "New", "York", "City" 三个 token 合并为 "New York City"
    aggregation_strategy="simple",
)

text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
entities = ner_pipeline(text)

print(f"文本: {text}")
print("识别到的实体：")
for entity in entities:
    print(f"  [{entity['entity_group']}] {entity['word']}  (置信度: {entity['score']:.4f})")
print()


# ─────────────────────────────────────────────
# 3. 抽取式问答（Question Answering）
#    典型应用：文档问答、FAQ 检索、合同信息抽取
#    输出：答案文本、在段落中的起止位置、置信度
#    注意：答案必须是段落中的原文片段（抽取式，非生成式）
# ─────────────────────────────────────────────

print("=" * 60)
print("3. 抽取式问答（Question Answering）")
print("=" * 60)

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

context = """
Transformers are deep learning models that use self-attention mechanisms to process sequential data.
They were introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google.
The architecture quickly became the dominant approach for natural language processing tasks,
and later expanded to computer vision, speech, and multimodal applications.
"""

questions = [
    "Who introduced the Transformer architecture?",
    "When was the Transformer paper published?",
    "What mechanism do Transformers use?",
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}  (置信度: {result['score']:.4f})")
    print()


# ─────────────────────────────────────────────
# 4. 文本摘要（Summarization）
#    典型应用：新闻摘要、长文档压缩、会议纪要生成
#    输出：比原文短的摘要文本（抽象式，不是直接截取原句）
# ─────────────────────────────────────────────

print("=" * 60)
print("4. 文本摘要（Summarization）")
print("=" * 60)

summarization_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

article = """
Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks.
These models, trained on massive text corpora, can generate coherent text, answer questions,
translate languages, and even write code. The development of models like GPT-3, GPT-4, and LLaMA
has accelerated research in artificial intelligence. However, these models also present challenges
related to computational cost, energy consumption, potential biases, and the spread of misinformation.
Researchers are actively working on techniques to make these models more efficient, accurate, and
safer for deployment in real-world applications.
"""

# max_length 和 min_length 控制摘要的字符数范围
result = summarization_pipeline(
    article,
    max_length=80,
    min_length=30,
    do_sample=False  # 确定性输出
)

print(f"原文（{len(article.split())} 词）:")
print(article.strip())
print(f"\n摘要（{len(result[0]['summary_text'].split())} 词）:")
print(result[0]['summary_text'])
print()


# ─────────────────────────────────────────────
# 5. 文本生成（Text Generation）
#    典型应用：内容创作辅助、代码补全、故事续写
#    输出：在 prompt 基础上继续生成的文本
# ─────────────────────────────────────────────

print("=" * 60)
print("5. 文本生成（Text Generation）")
print("=" * 60)

generation_pipeline = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = "The key to building reliable AI systems is"

# do_sample=True：采样解码（有随机性，每次结果可能不同）
# temperature=0.8：控制随机性，值越低输出越保守
# max_new_tokens：在 prompt 基础上最多生成多少个新 token
result = generation_pipeline(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    num_return_sequences=1,
    pad_token_id=generation_pipeline.tokenizer.eos_token_id,
)

print(f"Prompt: {prompt}")
print(f"生成结果:")
print(result[0]['generated_text'])
print()

print("=" * 60)
print("所有示例运行完毕")
print("=" * 60)
