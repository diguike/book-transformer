---
title: 第 12 章 实战三——TypeScript 集成方案
feishu_url: "https://www.feishu.cn/wiki/MWvFwv77riLekgktTLccAnmenCY"
last_synced: "2026-05-07T23:07:43"
---

## JS 生态里的 Transformer 工具链

Python 占据了机器学习工具链的主导地位，但大量线上服务跑在 Node.js 上。工程师面临的现实选择不是"要不要用 Transformer"，而是"用什么方式集成"。

**`@huggingface/transformers`**

这是 Hugging Face 官方维护的 JavaScript 库（前身是 `transformers.js`），本质上是 ONNX Runtime Web 的高层封装。它的工作方式是：先将 PyTorch 模型转换为 ONNX 格式，再通过 ONNX Runtime 执行推理。这个转换步骤由 Hugging Face 预先完成，用户直接下载转换好的 `.onnx` 文件即可。

```
Python 训练 (PyTorch)
       ↓
  ONNX 导出 (Hugging Face Hub 提供)
       ↓
ONNX Runtime Web / Node (JS 执行推理)
```

**`onnxruntime-node`**

`@huggingface/transformers` 在 Node.js 环境下内部依赖 `onnxruntime-node`。如果只需要推理自定义 ONNX 模型（不经过 Hugging Face Hub），可以直接使用 `onnxruntime-node`，控制粒度更细。

**与 Python 生态的差距**

| 维度 | Python (PyTorch) | JS (@huggingface/transformers) |
|------|-----------------|-------------------------------|
| 模型支持 | 全量 | 有限（需官方或社区提供 ONNX 版本） |
| 推理性能 | GPU 加速，成熟优化 | CPU 为主，GPU 支持有限 |
| 量化/优化 | 丰富（GPTQ、AWQ、bitsandbytes） | 基础量化（INT8/FP16） |
| 训练/微调 | 完整支持 | 不支持 |
| 生态成熟度 | 高 | 中等，快速发展 |

对于推理任务，`@huggingface/transformers` 覆盖了常见 NLP 场景（分类、NER、embedding、生成）。复杂的优化需求（如极低延迟的大模型推理）仍然需要 Python 侧的专用推理服务。

---

## 场景一：Node.js 后端推理

**适用条件：**

- 对延迟要求不极致（100ms 级别可以接受）
- 不想维护单独的 Python 推理服务
- 模型规模在几百 MB 以内（如 DistilBERT、MiniLM）
- 推理频率不高（峰值 QPS < 10）

**安装依赖**

```bash
npm install @huggingface/transformers
npm install -D typescript @types/node ts-node
```

**情感分析示例**

```typescript
// src/node_inference.ts
import { pipeline } from '@huggingface/transformers';

// 首次运行会下载模型到本地缓存（~/.cache/huggingface/hub）
// 模型大小约 67MB，之后复用缓存
const classifier = await pipeline(
  'text-classification',
  'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
);

const result = await classifier('This movie is absolutely fantastic!');
console.log(result);
// 输出: [{ label: 'POSITIVE', score: 0.9998 }]
```

**pipeline 支持的任务类型：**

| 任务类型字符串 | 用途 |
|--------------|------|
| `text-classification` | 文本分类、情感分析 |
| `token-classification` | NER、词性标注 |
| `feature-extraction` | 提取句子向量（embedding） |
| `text-generation` | 文本生成（小模型） |
| `translation` | 翻译 |
| `summarization` | 摘要 |
| `zero-shot-classification` | 零样本分类 |

**性能特点**

Node.js 推理使用 `onnxruntime-node`，默认跑在 CPU 上。对于 DistilBERT 规模的模型，单次推理耗时通常在 20-100ms（取决于序列长度和硬件）。首次推理还包含模型加载时间，热身后性能稳定。

如果推理频率高，可以将 `pipeline` 实例缓存在模块级别，避免重复加载：

```typescript
// 模块级缓存，进程生命周期内只初始化一次
let classifier: Awaited<ReturnType<typeof pipeline>> | null = null;

async function getClassifier() {
  if (!classifier) {
    classifier = await pipeline(
      'text-classification',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
  }
  return classifier;
}
```

完整可运行代码见 `examples/src/node_inference.ts`。

---

## 场景二：浏览器端推理

**适用条件：**

- 离线处理场景（PWA、桌面 Web 应用）
- 隐私敏感数据不能发送到服务器
- 只需要轻量任务（分类、嵌入）

**限制：**

- 只能用小模型：浏览器的内存和计算资源有限，一般使用 Tiny/Small 级别的模型（< 50MB）
- 首次加载慢：模型文件通过网络下载，需要做好缓存策略
- 不支持 GPU 计算（WebGPU 支持还在推进中）

**基本用法**

`@huggingface/transformers` 同时支持 Node.js 和浏览器环境。浏览器端的引入方式使用 ES Module：

```html
<script type="module">
  import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

  const classifier = await pipeline(
    'text-classification',
    'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
  );

  const result = await classifier(document.getElementById('input').value);
  console.log(result);
</script>
```

**配合 Web Worker 使用**

推理运算会阻塞主线程。生产环境中必须将推理放入 Web Worker：

```typescript
// inference.worker.ts
import { pipeline } from '@huggingface/transformers';

let classifier: Awaited<ReturnType<typeof pipeline>> | null = null;

self.onmessage = async (event: MessageEvent<{ text: string }>) => {
  if (!classifier) {
    classifier = await pipeline(
      'text-classification',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
  }

  const result = await classifier(event.data.text);
  self.postMessage(result);
};
```

```typescript
// main.ts
const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
  type: 'module',
});

worker.postMessage({ text: 'I really enjoyed this experience!' });
worker.onmessage = (event) => {
  console.log('推理结果:', event.data);
};
```

**模型缓存**

浏览器端模型文件通过 Cache API 缓存，存储在 Origin Private File System（OPFS）中。下载一次后，后续直接从本地加载，无需重复下载。

---

## 场景三：对接 Embedding API

当不需要在本地跑模型时，直接调用 API 是最轻量的方案。OpenAI 的 `text-embedding-3-small` 和 `text-embedding-3-large` 是目前综合性价比较高的选择。

**安装**

```bash
npm install openai
```

**获取文本向量**

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  // baseURL 可以替换为兼容 OpenAI 格式的本地服务：
  // baseURL: 'http://localhost:11434/v1'  // Ollama
  // baseURL: 'http://localhost:8000/v1'   // vLLM
});

const response = await client.embeddings.create({
  model: 'text-embedding-3-small',
  input: ['文本一', '文本二', '文本三'],
  encoding_format: 'float',
});

// response.data 是按 index 排序的向量列表
const embeddings = response.data.map((item) => item.embedding);
// embeddings[0] 是一个 1536 维的 number[]
```

**`text-embedding-3-small` 的关键参数：**

| 参数 | 值 |
|------|-----|
| 向量维度 | 1536（可通过 `dimensions` 参数压缩到 256-1536） |
| 最大输入长度 | 8191 tokens |
| 批量请求上限 | 2048 条/次 |
| 费用 | $0.02 / 1M tokens（截至 2024 年） |

**替代方案**

`openai` SDK 支持通过 `baseURL` 参数指向任何兼容 OpenAI API 格式的服务：

- **Ollama**（`http://localhost:11434/v1`）：本地运行开源模型，适合开发环境和数据不出境场景
- **vLLM**（`http://localhost:8000/v1`）：高性能推理服务，适合生产环境自部署
- **Groq、Together AI**：第三方托管服务，通常比 OpenAI 便宜

只需修改 `baseURL` 和 `model`，其余代码不变。

完整示例见 `examples/src/embedding_api.ts`。

---

## 三种方案的选型对比

| 维度 | Node.js 本地推理 | 浏览器端推理 | Embedding API |
|------|-----------------|-------------|---------------|
| **延迟** | 中（20-200ms） | 高（首次加载慢，推理 50-500ms） | 低（网络 RTT + API 处理，50-200ms） |
| **部署复杂度** | 低（无额外服务） | 低（纯前端） | 极低（无需部署模型） |
| **模型规模限制** | 中（< 500MB 合理） | 严格（< 50MB 为宜） | 无限制（由 API 提供商决定） |
| **可用性依赖** | 无外部依赖 | 无外部依赖 | 依赖 API 服务 | 
| **数据隐私** | 数据不出机器 | 数据不离开浏览器 | 数据发送给 API 提供商 |
| **成本** | 计算成本（服务器 CPU） | 用户设备算力 | API 调用费用 |
| **适用场景** | 低频推理、不想维护 Python 服务 | 离线、隐私敏感场景 | 大多数生产场景 |

**决策流程：**

```
数据是否涉及隐私 / 需要离线使用？
  ├── 是 → 本地推理（Node.js 或浏览器端）
  │         └── 是否在浏览器中？
  │               ├── 是 → 浏览器端推理（限 < 50MB 模型）
  │               └── 否 → Node.js 本地推理
  └── 否 → Embedding API（首选，最简单）
             └── 需要控制成本或数据主权？
                   └── 是 → 自部署 Ollama/vLLM + OpenAI 兼容 API
```

---

## 实战：给现有 Web 项目加语义搜索能力

这一节将上面的工具拼在一起，实现一个完整的语义搜索功能。场景是为一个知识库加上"相关文档推荐"。

**架构**

```
用户查询
    ↓
embed(query)          ← OpenAI Embedding API
    ↓
cosineSimilarity()    ← 纯 JS 计算，内存操作
    ↓
排序，返回 topK        ← 无外部数据库
    ↓
相关文档列表
```

这个实现不依赖向量数据库，文档列表存在内存里。文档数量在数千条以内，这种方式完全够用，部署复杂度极低。

**核心接口**

```typescript
// 构建内存索引
const index = await buildIndex(documents: string[])

// 执行语义搜索
const results = await search(index, query: string, topK = 3)
// 返回: { document: string; score: number; rank: number }[]
```

**`buildIndex` 实现要点**

- 分批调用 API，每批 100 条，避免超出请求体积限制
- 返回 `IndexEntry[]`，每条记录存储原文和对应的向量

**`search` 实现要点**

- 对查询文本调用一次 Embedding API
- 遍历所有 IndexEntry，计算余弦相似度
- 按相似度降序排序，取前 topK 条

**接入现有项目**

以 Express.js 接口为例：

```typescript
import { buildIndex, search } from './semantic_search.js';

// 应用启动时构建索引（一次性操作）
const docs = await loadDocumentsFromDB();  // 从数据库加载文档
const index = await buildIndex(docs);

// 搜索接口
app.get('/search', async (req, res) => {
  const { q, k = 5 } = req.query;

  if (typeof q !== 'string') {
    return res.status(400).json({ error: 'missing query' });
  }

  const results = await search(index, q, Number(k));
  res.json(results);
});
```

**扩展方向**

文档数量超过 1 万条后，线性扫描的性能会下降。此时可以切换到向量数据库：

- **pgvector**：PostgreSQL 插件，无需引入新基础设施，适合已有 Postgres 的项目
- **Qdrant**：独立向量数据库，支持过滤、分片，适合向量检索是核心功能的场景
- **Chroma**：轻量嵌入式向量数据库，适合单机或开发环境

切换时，只需替换 `buildIndex` 和 `search` 的实现，上层接口保持不变。

完整的 `semantic_search.ts` 实现见 `examples/src/semantic_search.ts`。
