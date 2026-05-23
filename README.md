# Transformer 工程实战

> 在线阅读 · [inferloop.dev/transformer](https://inferloop.dev/transformer)  
> 所有书目 · [inferloop.dev](https://inferloop.dev)

面向有工程背景、希望转型 AI 方向的工程师。不管你是前端、客户端还是后端，只要写过几年代码，这本书就是为你写的。

## 在线阅读

本书在线版：**[inferloop.dev](https://inferloop.dev)**

内容持续更新，在线版包含最新勘误。

## 目标

读完这本书，你能：

- 理解 Transformer 的工作原理，能向别人解释清楚
- 用 HuggingFace 生态完成文本分类、语义搜索、RAG 等真实任务
- 跑通三个完整的实战项目，代码可直接 clone 运行
- 在需要时用 TypeScript 接入 AI 能力

## 阅读前提

- 有 3 年以上工程经验，能读懂代码
- 会 JavaScript / TypeScript（Python 不熟没关系，第 0 章补）
- 高中数学水平（矩阵乘法会用就行，不需要推导）

## 代码运行环境

- Python 3.10+
- Node.js 18+（第 12 章）
- 第 8 章微调需要 GPU，推荐阿里云 / 火山云按量付费实例

## 目录结构

```
transformer-book/
  00-python-quickstart/    # 第 0 章
  01-text-to-vector/       # 第 1 章
  02-attention/            # 第 2 章
  03-multi-head-attention/ # 第 3 章
  04-transformer-arch/     # 第 4 章
  05-arch-variants/        # 第 5 章
  06-huggingface/          # 第 6 章
  07-embeddings/           # 第 7 章
  08-finetuning/           # 第 8 章
  09-inference/            # 第 9 章
  10-semantic-search/      # 第 10 章（实战项目）
  11-rag/                  # 第 11 章（实战项目）
  12-typescript/           # 第 12 章（实战项目）
  appendix/                # 附录
```

每章的 `examples/` 目录包含可独立运行的完整代码。

## 运行示例

```bash
cd 02-attention/examples
pip install -r requirements.txt
python attention_scratch.py
```

## 许可证

MIT


## 相关书

来自同一作者的其他书:

- [《Hermes Agent 源码解读》](https://inferloop.dev/hermes-agent)
- [《LLM Infra 工程实战》](https://inferloop.dev/llm-infra)
- [《AI Token 中转站实战》](https://inferloop.dev/llm-gateway)
- [《Agent Memory 工程实战》](https://inferloop.dev/claude-mem)
- [《百万级 AI Agent 平台架构》](https://inferloop.dev/enterprise-agent)
- [《OpenClaw 源码解析》](https://inferloop.dev/openclaw)
- [《Claude Code Skill 开发指南》](https://inferloop.dev/claude-skill)
- [《Claude 插件官方指南》](https://inferloop.dev/claude-plugins)
- [《自己动手写 AI Agent》](https://inferloop.dev/ling-agent)
