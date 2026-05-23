---
title: 附录 B 常用模型选型速查表
feishu_url: "https://www.feishu.cn/wiki/RyHiwacV4iGOV3kdWNIcPiOTnwg"
last_synced: "2026-05-07T23:07:58"
---

## 按任务类型速查

表格中的参数量为近似值。HuggingFace 模型名可直接传入 `from_pretrained()` 或 `pipeline()`。

### 文本分类

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 文本分类（英文） | DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | 66M | 速度快，适合延迟敏感场景 |
| 文本分类（英文） | BERT base | `textattack/bert-base-uncased-ag-news` | 110M | 精度更高，可在 AG News 等数据集上继续微调 |
| 文本分类（中文） | BERT 中文 | `bert-base-chinese` | 102M | 中文 NLP 的基础底座，微调起点 |
| 文本分类（中文） | RoBERTa 中文 | `hfl/chinese-roberta-wwm-ext` | 102M | 哈工大开源，中文效果优于官方 BERT |
| 零样本分类 | BART MNLI | `facebook/bart-large-mnli` | 407M | 无需标注数据，直接用候选标签分类 |

### 情感分析

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 情感分析（英文，二分类） | DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` | 66M | POSITIVE/NEGATIVE，最常用的英文情感模型 |
| 情感分析（英文，多维度） | Twitter RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 125M | 针对社交媒体文本优化 |
| 情感分析（中文） | RoBERTa 情感 | `uer/roberta-base-finetuned-jd-binary-chinese` | 102M | 在京东评论数据集上微调，适合电商场景 |
| 情感分析（多语言） | XLM-RoBERTa | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 278M | 支持 100+ 语言 |

### 命名实体识别（NER）

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| NER（英文） | BERT CoNLL-2003 | `dbmdz/bert-large-cased-finetuned-conll03-english` | 335M | 标准 NER 基准，识别人名/地名/组织名 |
| NER（英文，轻量） | DistilBERT NER | `elastic/distilbert-base-uncased-finetuned-conll03-english` | 66M | 速度优先时使用 |
| NER（中文） | BERT 中文 NER | `hfl/chinese-bert-wwm-ext` + 自定义微调 | 102M | 需在中文 NER 数据集上微调（如 MSRA、人民日报） |
| NER（多语言） | XLM-RoBERTa NER | `xlm-roberta-large-finetuned-conll03-english` | 560M | 多语言场景 |

### 文本相似度

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 语义相似度（英文） | SBERT MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 22M | 速度极快，精度够用，入门首选 |
| 语义相似度（英文，高精度） | SBERT MPNet | `sentence-transformers/all-mpnet-base-v2` | 109M | 精度更高，适合精度优先场景 |
| 语义相似度（中英文） | paraphrase-MiniLM | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | 支持 50+ 语言 |
| 语义相似度（中文） | text2vec | `shibing624/text2vec-base-chinese` | 102M | 中文语义相似度专用模型 |

### 语义搜索 Embedding

| 任务类型 | 推荐模型 | HuggingFace 模型名 / API | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 语义搜索（本地部署） | BGE-M3 | `BAAI/bge-m3` | 568M | 多语言，多粒度检索，目前综合性最强的开源 embedding 模型 |
| 语义搜索（本地，轻量） | BGE small | `BAAI/bge-small-en-v1.5` | 33M | 英文，速度快 |
| 语义搜索（本地，中文） | BGE large 中文 | `BAAI/bge-large-zh-v1.5` | 326M | 中文检索推荐 |
| 语义搜索（API） | text-embedding-3-small | OpenAI API | — | 1536 维，$0.02/1M tokens，性价比高 |
| 语义搜索（API） | text-embedding-3-large | OpenAI API | — | 3072 维，精度更高，$0.13/1M tokens |

### 文本摘要

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 摘要（英文） | BART large CNN | `facebook/bart-large-cnn` | 407M | 在 CNN/DailyMail 上微调，新闻摘要效果好 |
| 摘要（英文，轻量） | DistilBART | `sshleifer/distilbart-cnn-12-6` | 306M | BART 的蒸馏版 |
| 摘要（中文） | mT5 | `csebuetnlp/mT5_multilingual_XLSum` | 300M | 支持多语言摘要，含中文 |
| 摘要（长文档） | LongT5 | `google/long-t5-tglobal-base` | 248M | 处理长文档（> 512 tokens） |

### 机器翻译

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 翻译（英→中） | Helsinki-NLP | `Helsinki-NLP/opus-mt-en-zh` | 74M | 轻量，适合大批量翻译 |
| 翻译（中→英） | Helsinki-NLP | `Helsinki-NLP/opus-mt-zh-en` | 74M | 同上，反向 |
| 翻译（多语言） | NLLB-200 | `facebook/nllb-200-distilled-600M` | 600M | 支持 200 种语言互译 |
| 翻译（高质量） | M2M-100 | `facebook/m2m100_1.2B` | 1.2B | 高精度多语言翻译 |

### 文本生成与对话

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 文本生成（英文，小） | GPT-2 | `gpt2` | 124M | 教学和原型验证用，效果有限 |
| 文本生成（英文，中等） | GPT-2 XL | `gpt2-xl` | 1.5B | 效果明显好于 GPT-2 |
| 指令跟随（轻量） | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 本地部署的最小可用指令模型 |
| 指令跟随（平衡） | Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | 3B | 需申请访问权限 |
| 对话（本地，主力） | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | 需要约 16GB 显存（FP16） |

### 代码生成

| 任务类型 | 推荐模型 | HuggingFace 模型名 | 参数量 | 备注 |
|---------|---------|-------------------|-------|------|
| 代码补全（轻量） | CodeBERT | `microsoft/codebert-base` | 125M | 代码理解，适合做 embedding 和分类 |
| 代码生成（小） | Qwen2.5-Coder-1.5B | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | 代码专用，可本地运行 |
| 代码生成（主力） | Qwen2.5-Coder-7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | 综合代码能力强，开源模型中表现突出 |
| 代码生成（高端） | DeepSeek-Coder-V2 | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B | 需较大显存 |

---

## 选型建议

**从任务出发：**

1. 先明确任务类型和语言（中文/英文/多语言）
2. 确定部署方式（本地推理 vs API）
3. 在精度和速度之间取舍

**从规模出发：**

- 实验/原型：优先选 < 200M 参数的模型，迭代快
- 生产环境：根据延迟和硬件预算确定上限
- 7B 是目前"效果够用 + 可自部署"的主流选择，需要约 16GB 显存

**从语言出发：**

- 中文任务：BGE 系列（embedding）、哈工大 HFL 系列（分类/NER）、Qwen 系列（生成）
- 英文任务：Sentence-Transformers、BERT/RoBERTa 系列
- 多语言：XLM-RoBERTa、NLLB-200、BGE-M3

---

> 本附录来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
