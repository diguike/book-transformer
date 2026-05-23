/**
 * Embedding API 调用示例
 *
 * 使用 OpenAI SDK 调用 text-embedding-3-small 模型，
 * 对文本做向量化，并计算余弦相似度。
 *
 * 环境变量：
 *   OPENAI_API_KEY  - OpenAI API 密钥（必填）
 *   OPENAI_BASE_URL - API 地址（可选，默认为 OpenAI 官方）
 *                     可替换为兼容 OpenAI 格式的本地服务，例如：
 *                     http://localhost:11434/v1  （Ollama）
 *                     http://localhost:8000/v1   （vLLM）
 *
 * 运行方式：npm run embedding-api
 */

import OpenAI from 'openai';

// 从环境变量读取配置
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('错误：未设置 OPENAI_API_KEY 环境变量');
  process.exit(1);
}

const client = new OpenAI({
  apiKey,
  // baseURL 为可选项，留空使用官方端点
  // baseURL: process.env.OPENAI_BASE_URL,
});

// 待向量化的文本片段
const documents = [
  '机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。',
  '深度学习使用多层神经网络来学习数据的层次化表示。',
  'Transformer 架构彻底改变了自然语言处理领域。',
  '今天天气很好，适合出去散步。',
  '股票市场今天表现强劲，科技板块领涨。',
];

/**
 * 调用 OpenAI Embedding API，返回向量列表
 * @param texts 文本列表
 * @param model 模型名称
 */
async function getEmbeddings(
  texts: string[],
  model = 'text-embedding-3-small'
): Promise<number[][]> {
  const response = await client.embeddings.create({
    model,
    input: texts,
    // encoding_format 默认为 float，也可用 base64 节省传输体积
    encoding_format: 'float',
  });

  // 按原始顺序返回向量（API 返回的 data 数组已按 index 排序）
  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}

/**
 * 计算两个向量的余弦相似度
 * 余弦相似度范围 [-1, 1]，1 表示完全相同方向，0 表示正交，-1 表示相反
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] ** 2;
    normB += vecB[i] ** 2;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main(): Promise<void> {
  console.log('正在获取文本向量...\n');

  const embeddings = await getEmbeddings(documents);
  console.log(`向量维度：${embeddings[0].length}\n`);

  console.log('='.repeat(60));
  console.log('文档两两相似度：');
  console.log('='.repeat(60));

  // 计算所有文档对之间的相似度
  for (let i = 0; i < documents.length; i++) {
    for (let j = i + 1; j < documents.length; j++) {
      const sim = cosineSimilarity(embeddings[i], embeddings[j]);
      console.log(`\n文档 ${i + 1}: "${documents[i].slice(0, 20)}..."`);
      console.log(`文档 ${j + 1}: "${documents[j].slice(0, 20)}..."`);
      console.log(`相似度: ${sim.toFixed(4)}`);
    }
  }
}

main().catch((err) => {
  console.error('调用失败:', err);
  process.exit(1);
});
