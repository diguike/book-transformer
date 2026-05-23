/**
 * Node.js 后端推理示例
 *
 * 使用 @huggingface/transformers 在 Node.js 中做情感分析。
 * 首次运行时会自动下载模型到本地缓存（~/.cache/huggingface/hub），
 * 约 67MB，之后复用缓存无需重复下载。
 *
 * 运行方式：npm run node-inference
 */

import { pipeline } from '@huggingface/transformers';

// 待分类的句子列表
const sentences = [
  'This product is absolutely fantastic, I love it!',
  'Terrible experience, completely disappointed.',
  'It works fine, nothing special.',
  'Best purchase I have made this year.',
  'The documentation is outdated and confusing.',
];

async function runSentimentAnalysis(): Promise<void> {
  console.log('正在加载模型...');
  console.log('首次运行会下载模型到本地缓存（约 67MB），请耐心等待\n');

  // 加载情感分析 pipeline
  // 模型：distilbert-base-uncased，在 SST-2 数据集上微调
  // ONNX Runtime 会根据当前环境自动选择 CPU 后端
  const classifier = await pipeline(
    'text-classification',
    'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
  );

  console.log('模型加载完成，开始推理\n');
  console.log('='.repeat(60));

  for (const sentence of sentences) {
    const startTime = Date.now();

    // pipeline 返回数组，每个元素对应输入序列的结果
    const result = await classifier(sentence);
    const elapsed = Date.now() - startTime;

    // 取第一个结果（单句输入）
    const output = Array.isArray(result) ? result[0] : result;

    console.log(`输入: ${sentence}`);
    console.log(
      `结果: ${(output as { label: string; score: number }).label} ` +
        `(置信度: ${((output as { label: string; score: number }).score * 100).toFixed(1)}%)`
    );
    console.log(`耗时: ${elapsed}ms`);
    console.log('-'.repeat(60));
  }
}

runSentimentAnalysis().catch((err) => {
  console.error('推理失败:', err);
  process.exit(1);
});
