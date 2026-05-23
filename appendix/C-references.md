---
title: 附录 C 参考资料与延伸阅读
feishu_url: "https://www.feishu.cn/wiki/ZXp6wnyhBiwF1nkLnGtcKyqLnUg"
last_synced: "2026-05-07T23:08:06"
---

## 原始论文

| 论文 | 作者/来源 | 链接 | 适合什么时候读 |
|------|---------|------|--------------|
| Attention Is All You Need | Vaswani et al., Google, 2017 | https://arxiv.org/abs/1706.03762 | 读完第 3-4 章后，对照原文验证自己的理解 |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al., Google, 2018 | https://arxiv.org/abs/1810.04805 | 学完 HuggingFace 生态后，理解 BERT 为何成为 NLP 里程碑 |
| Language Models are Unsupervised Multitask Learners（GPT-2） | Radford et al., OpenAI, 2019 | https://openai.com/research/language-unsupervised | 理解 GPT 路线时，与 BERT 路线对比阅读 |
| Improving Language Understanding by Generative Pre-Training（GPT-1） | Radford et al., OpenAI, 2018 | https://openai.com/research/language-unsupervised | 了解预训练 + 微调范式的起点 |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al., Facebook, 2019 | https://arxiv.org/abs/1907.11692 | 看 BERT 如何通过训练细节大幅提升性能 |
| Exploring the Limits of Transfer Learning with T5 | Raffel et al., Google, 2019 | https://arxiv.org/abs/1910.10683 | 理解 encoder-decoder 架构和"文本到文本"范式 |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al., Microsoft, 2021 | https://arxiv.org/abs/2106.09685 | 准备做微调时，读这篇先理解 LoRA 的核心思路 |

---

## 可视化教程：Jay Alammar 系列

Jay Alammar 的博客是目前最好的 Transformer 可视化讲解，配图精准，适合和本书配合阅读。

| 文章 | 链接 | 适合什么时候读 |
|------|------|--------------|
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | 读完第 4 章后，用这篇文章的配图复习一遍完整架构 |
| The Illustrated BERT, ELMo, and co. | https://jalammar.github.io/illustrated-bert/ | 读完 BERT 论文前后，建立直觉 |
| The Illustrated GPT-2 | https://jalammar.github.io/illustrated-gpt2/ | 对比 BERT 之后，理解 GPT 的自回归生成机制 |
| Visualizing A Neural Machine Translation Model (Attention) | https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ | 读第 2 章（Attention 机制）之前，先看这篇建立直觉 |
| A Visual Guide to Using BERT for the First Time | https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/ | 刚开始用 HuggingFace 时，跟着这篇操作一遍 |

---

## 开源实现参考

这几个项目的代码量适中，适合工程师阅读和学习。

| 项目 | 链接 | 适合什么时候读 |
|------|------|--------------|
| happy-llm | https://github.com/datawhalechina/happy-llm | 想用中文材料从头实现 LLM 时，这是目前最好的中文开源教材配套代码 |
| LLMs-from-scratch | https://github.com/rasbt/LLMs-from-scratch | 想用 PyTorch 从零构建 GPT 风格模型时，按章节跟代码实现 |
| minimind | https://github.com/jingyaogong/minimind | 想在一台普通机器上跑通 LLM 完整训练流程（预训练+微调）时，用这个项目做实验 |
| nanoGPT | https://github.com/karpathy/nanoGPT | 看 Andrej Karpathy 写的极简 GPT 实现，理解训练循环和模型结构 |
| minbpe | https://github.com/karpathy/minbpe | 理解 BPE tokenizer 实现细节时，读这个约 500 行的极简实现 |

---

## HuggingFace 官方文档

| 文档 | 链接 | 适合什么时候读 |
|------|------|--------------|
| Transformers 文档首页 | https://huggingface.co/docs/transformers | 当需要查某个 API 用法时，从这里开始 |
| pipeline API | https://huggingface.co/docs/transformers/main_classes/pipelines | 第 6 章后，用 pipeline 时遇到不熟悉的参数就查这里 |
| Trainer API | https://huggingface.co/docs/transformers/main_classes/trainer | 准备做微调时，先通读一遍 Trainer 的参数 |
| PEFT（Parameter-Efficient Fine-Tuning） | https://huggingface.co/docs/peft | LoRA/QLoRA 微调的官方工具库文档，微调章节的必读配套 |
| Datasets 文档 | https://huggingface.co/docs/datasets | 处理训练数据时，查 Dataset 的加载、过滤、映射操作 |
| Accelerate 文档 | https://huggingface.co/docs/accelerate | 需要多卡训练或混合精度时参考 |
| Tokenizers 文档 | https://huggingface.co/docs/tokenizers | 需要深入理解或自定义 tokenizer 时 |

---

## 推荐的进阶方向

读完本书后，以下方向可以继续深入。

### LLM 推理优化

| 资料 | 链接 | 说明 |
|------|------|------|
| vLLM 论文（PagedAttention） | https://arxiv.org/abs/2309.06180 | KV Cache 管理的核心创新，理解高吞吐推理服务的基础 |
| FlashAttention 论文 | https://arxiv.org/abs/2205.14135 | IO 感知的高效 Attention 实现，理解为什么现代推理框架都用它 |
| Speculative Decoding | https://arxiv.org/abs/2211.17192 | 用小模型加速大模型生成，重要的推理加速技术 |
| llm.c（Karpathy） | https://github.com/karpathy/llm.c | 纯 C 实现的 GPT-2 训练，理解底层计算细节 |

### RAG 进阶

| 资料 | 链接 | 说明 |
|------|------|------|
| RAG 原始论文 | https://arxiv.org/abs/2005.11401 | Lewis et al., Facebook 2020，RAG 的出处 |
| Advanced RAG（LlamaIndex 博客） | https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-rag-cheat-sheet-2852e0bbadba | 覆盖 Hybrid Search、Reranking、HyDE 等进阶技术 |
| RAGAS（RAG 评估框架） | https://github.com/explodinggradients/ragas | 构建 RAG 系统后，用这个框架评估检索和生成质量 |

### Fine-tuning 进阶

| 资料 | 链接 | 说明 |
|------|------|------|
| QLoRA 论文 | https://arxiv.org/abs/2305.14314 | 4-bit 量化 + LoRA，消费级 GPU 上微调 65B 模型的方案 |
| RLHF 综述 | https://arxiv.org/abs/2203.02155 | InstructGPT 论文，理解 RLHF 对齐训练的完整流程 |
| DPO 论文 | https://arxiv.org/abs/2305.18290 | Direct Preference Optimization，RLHF 的简化替代方案 |
| Axolotl | https://github.com/axolotl-ai-cloud/axolotl | 工程化的微调框架，支持 LoRA/QLoRA/全量微调，配置驱动 |

---

> 本附录来自《Transformer 工程实战》开源版 · 作者「递归客」  
> 在线阅读完整书系：[inferloop.dev](https://inferloop.dev)  
> 源码仓库：[github.com/diguike/book-transformer](https://github.com/diguike/book-transformer)
