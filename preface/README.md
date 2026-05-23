---
title: 前言
feishu_url: "https://www.feishu.cn/wiki/GwN4wanrCikvfGk8bLgcdgVentb"
last_synced: "2026-05-07T23:06:09"
---

## 这本书是为谁写的

如果你有三到五年的工程经验，写过前端、后端或客户端代码，现在想搞清楚 LLM 到底是怎么工作的——这本书是为你写的。

不是为了研究员，不是为了刚入行的新手，也不是为了只想调 API 就完事的人。

本书假设你不懂 Python，也没有机器学习基础，第 0 章提供了足够的 Python 入门内容。

市面上关于 Transformer 的资料要么是论文级别的数学推导，要么是"5分钟学会调用 ChatGPT"。这两类内容中间有一个巨大的空白：工程师需要理解底层机制，但不需要手推梯度。这本书填的就是这个空白。

## 为什么要读懂 Transformer，而不只是调 API

调 API 确实能完成很多任务。但工程决策层面的问题，光靠调 API 解决不了：

- 为什么这个 Embedding 模型在我的场景里效果差？换一个能解决问题吗？
- RAG 的检索质量不好，瓶颈在向量化还是在检索策略？
- 微调 vs 直接调用 API，在这个场景下该选哪个？成本和适用场景分别是什么？
- 为什么上下文窗口越长，推理越慢？这个限制是根本的吗？

这些问题的答案都在 Transformer 的架构里。不理解 Attention 机制，不知道 token 是怎么被处理的，做架构决策只能靠猜。

读完这本书，你不会变成 AI 研究员，但你能做出有根据的工程判断。

## 这本书覆盖什么，不覆盖什么

**覆盖：**
- Transformer 架构的核心机制：token 化、Embedding、Attention、Multi-Head Attention、完整的 Encoder-Decoder 结构
- 三种主要架构变体（Encoder-only、Decoder-only、Encoder-Decoder）以及它们各自适合的任务
- HuggingFace 生态的工程用法
- Embedding 的工程实践：相似度计算、向量存储
- 微调的基本流程
- 推理工程：批处理、延迟优化的基本思路
- 三个完整实战项目：语义搜索、RAG、TypeScript 集成

**不覆盖：**
- 数学推导（反向传播、梯度计算、损失函数的数学细节）
- 从头训练大模型
- 模型量化、分布式训练等高级推理工程话题
- 具体的云平台部署方案
- Prompt Engineering（提示词工程）：system/user/assistant 消息结构、few-shot、chain-of-thought 等技巧不在本书范围内

## 怎么读这本书

**第 0 章**是 Python 速成，如果你已经用过 Python，跳过。如果你是纯 JS/TS 背景，建议先跑一遍代码，确认环境正常。

**第 1-5 章**是核心。这五章按顺序讲 Transformer 的每一层，从最基础的"文字怎么变成数字"开始，一步步到完整架构。这部分建议顺序读，不要跳。

**第 6-9 章**是工具和工程实践。第 6 章 HuggingFace 是后续一切的基础，建议读。第 7-9 章（Embedding、微调、推理）可以按需跳读，和你当前工作最相关的优先。

**第 10-12 章**是三个实战项目。语义搜索、RAG、TypeScript 集成，可以直接跳到你最需要的那个。每个项目都是独立可运行的，代码在对应章节的 `examples/` 目录下。

示例代码可在各章 `examples/` 目录下运行，依赖已在 `requirements.txt` 中列出。Python 为主，关键处附 TypeScript 对照，方便 JS 背景的读者建立对应关系。

本书示例代码基于以下版本验证：
- Python 3.10+
- transformers 4.40.0
- torch 2.3.0
- sentence-transformers 2.7.0

如果运行时遇到 API 不兼容的报错，通常是库版本差异导致的。建议按照各章 `examples/requirements.txt` 安装指定版本。

本书在线版同步发布于 [inferloop.dev](https://inferloop.dev)，包含勘误更新和配套资源。如发现内容被转载，欢迎通过原地址核实最新版本。
