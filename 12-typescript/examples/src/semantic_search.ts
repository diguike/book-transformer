/**
 * 最小语义搜索实现
 *
 * 纯内存实现，不依赖任何外部数据库。
 * 适合文档数量在几千条以内的场景，超出这个规模应使用向量数据库（Qdrant、pgvector 等）。
 *
 * 用法示例见文件底部的 main 函数。
 */

import OpenAI from 'openai';

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('错误：未设置 OPENAI_API_KEY 环境变量');
  process.exit(1);
}

const client = new OpenAI({ apiKey });

// 向量索引中存储的单条记录
interface IndexEntry {
  document: string;
  embedding: number[];
}

// 搜索结果
interface SearchResult {
  document: string;
  score: number;
  rank: number;
}

/**
 * 调用 Embedding API 对单条或多条文本做向量化
 */
async function embed(
  texts: string | string[],
  model = 'text-embedding-3-small'
): Promise<number[][]> {
  const input = Array.isArray(texts) ? texts : [texts];

  const response = await client.embeddings.create({
    model,
    input,
    encoding_format: 'float',
  });

  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}

/**
 * 计算余弦相似度
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * 对文档列表做向量化，构建内存索引
 *
 * @param documents 文档字符串列表
 * @returns 内存索引（数组）
 *
 * 注意：文档数量较多时，建议分批调用 API（每批 100 条左右），
 * 避免单次请求体积过大。
 */
export async function buildIndex(documents: string[]): Promise<IndexEntry[]> {
  console.log(`正在为 ${documents.length} 条文档构建索引...`);

  // 分批处理，每批 100 条
  const batchSize = 100;
  const allEmbeddings: number[][] = [];

  for (let i = 0; i < documents.length; i += batchSize) {
    const batch = documents.slice(i, i + batchSize);
    const embeddings = await embed(batch);
    allEmbeddings.push(...embeddings);
    console.log(`  已处理 ${Math.min(i + batchSize, documents.length)} / ${documents.length}`);
  }

  const index: IndexEntry[] = documents.map((doc, i) => ({
    document: doc,
    embedding: allEmbeddings[i],
  }));

  console.log('索引构建完成\n');
  return index;
}

/**
 * 在已构建的索引中执行语义搜索
 *
 * @param index    由 buildIndex 返回的内存索引
 * @param query    查询字符串
 * @param topK     返回最相关的 K 条文档，默认为 3
 * @returns        按相似度降序排列的结果列表
 */
export async function search(
  index: IndexEntry[],
  query: string,
  topK = 3
): Promise<SearchResult[]> {
  // 对查询做向量化
  const [queryEmbedding] = await embed(query);

  // 计算查询向量与所有文档向量的相似度
  const scored = index.map((entry) => ({
    document: entry.document,
    score: cosineSimilarity(queryEmbedding, entry.embedding),
  }));

  // 按相似度降序排序，取前 topK 条
  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, topK).map((item, rank) => ({
    document: item.document,
    score: item.score,
    rank: rank + 1,
  }));
}

// ── 使用示例 ──────────────────────────────────────────────────

async function main(): Promise<void> {
  // 模拟知识库文档（实际场景可从文件/数据库加载）
  const documents = [
    'Transformer 模型由 Google 在 2017 年的论文 "Attention Is All You Need" 中提出。',
    'BERT 是一种双向 Transformer 编码器，在大量无标注文本上预训练而来。',
    'GPT 系列模型使用 Transformer 解码器架构，专注于文本生成任务。',
    '向量数据库用于存储和检索高维向量，是 RAG 系统的核心组件。',
    'RAG（检索增强生成）将检索系统与生成模型结合，减少幻觉问题。',
    '余弦相似度衡量两个向量之间的夹角，常用于衡量语义相似程度。',
    'fine-tuning 通过在特定任务数据上继续训练，使预训练模型适应特定场景。',
    'LoRA 是一种高效微调方法，只训练少量低秩矩阵，大幅降低显存需求。',
    'Node.js 是基于 V8 引擎的 JavaScript 运行时，适合构建高并发服务。',
    'TypeScript 在 JavaScript 基础上增加了静态类型系统，提升代码可维护性。',
  ];

  // 构建索引
  const index = await buildIndex(documents);

  // 执行几组查询
  const queries = [
    'Transformer 是怎么来的？',
    '如何降低大模型的微调成本？',
    'JavaScript 和 TypeScript 有什么关系？',
  ];

  for (const query of queries) {
    console.log(`查询："${query}"`);
    console.log('-'.repeat(50));

    const results = await search(index, query, 3);

    for (const result of results) {
      console.log(`#${result.rank} [${result.score.toFixed(4)}] ${result.document}`);
    }

    console.log('\n');
  }
}

main().catch((err) => {
  console.error('执行失败:', err);
  process.exit(1);
});
