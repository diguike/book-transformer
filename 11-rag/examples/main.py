"""
main.py — RAG 文档问答系统命令行入口

运行方式：
    python main.py --doc ./sample.txt

首次运行会对文档分块并建立向量索引。
后续运行如果文档未变，自动跳过索引构建步骤。
"""

from __future__ import annotations

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from chunker import chunk_text
from generator import generate_answer
from retriever import collection_count, index_chunks, retrieve

console = Console()

# 向量库路径（与 retriever.py 中 CHROMA_PATH 保持一致）
CHROMA_PATH = "./chroma_db"


def load_and_index(doc_path: str) -> int:
    """
    读取文档文件，分块，建立向量索引。

    Args:
        doc_path: 文档文件路径

    Returns:
        文本块数量
    """
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=512, overlap=64)
    source_name = os.path.basename(doc_path)

    console.print(f"[dim]文档共 {len(text)} 字符，切分为 {len(chunks)} 个文本块[/dim]")
    console.print("[dim]构建向量索引...[/dim]")

    index_chunks(chunks, source_name)
    return len(chunks)


def print_answer(answer: str, contexts: list[dict]) -> None:
    """用 Rich Panel 格式化输出回答和引用来源。"""
    # 主回答
    console.print(Panel(
        answer,
        title="[bold white]回答[/bold white]",
        border_style="green",
        padding=(1, 2),
    ))

    # 引用来源
    ref_lines = []
    for i, ctx in enumerate(contexts, start=1):
        score_color = "green" if ctx["score"] >= 0.6 else ("yellow" if ctx["score"] >= 0.4 else "red")
        score_text = f"[{score_color}]{ctx['score']:.4f}[/{score_color}]"

        # 截断长文本，只展示前 200 字符
        snippet = ctx["text"]
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        ref_lines.append(
            f"[{i}] 相似度: {score_text}  来源: [dim]{ctx['source']}[/dim]\n"
            f"    [dim]{snippet}[/dim]"
        )

    refs_content = "\n\n".join(ref_lines)
    console.print(Panel(
        Text.from_markup(refs_content),
        title="[bold white]引用来源[/bold white]",
        border_style="dim",
        padding=(1, 2),
    ))


def main():
    parser = argparse.ArgumentParser(description="基于 RAG 的文档问答系统")
    parser.add_argument(
        "--doc",
        type=str,
        default="./sample.txt",
        help="文档文件路径（默认：./sample.txt）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="检索的文本块数量（默认：4）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建索引（默认：已有索引则跳过）",
    )
    args = parser.parse_args()

    # 检查文档文件是否存在
    if not os.path.exists(args.doc):
        console.print(f"[red]错误：文件不存在 {args.doc}[/red]")
        raise SystemExit(1)

    # 判断是否需要构建索引
    need_index = args.rebuild or not os.path.exists(CHROMA_PATH)

    if need_index:
        chunk_count = load_and_index(args.doc)
        console.print(f"[green]索引构建完成，共 {chunk_count} 个文本块[/green]\n")
    else:
        count = collection_count()
        console.print(f"[dim]加载已有索引，共 {count} 个文本块（如需重建请加 --rebuild）[/dim]\n")

    console.print("[bold cyan]RAG 文档问答系统已就绪[/bold cyan]")
    console.print(f"[dim]文档：{args.doc}  检索 top-{args.top_k} 块[/dim]")
    console.print("[dim]输入问题后按 Enter，输入 q 退出[/dim]\n")

    # 交互式问答循环
    while True:
        try:
            question = console.input("[bold]问题> [/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]已退出[/dim]")
            break

        if not question:
            continue
        if question.lower() == "q":
            console.print("[dim]已退出[/dim]")
            break

        # 检索相关文本块
        contexts = retrieve(question, top_k=args.top_k)

        # 调用 LLM 生成回答
        with console.status("[dim]生成回答中...[/dim]"):
            try:
                answer = generate_answer(question, contexts)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                break
            except Exception as e:
                console.print(f"[red]调用 LLM 失败：{e}[/red]")
                continue

        print_answer(answer, contexts)
        console.print()


if __name__ == "__main__":
    main()
