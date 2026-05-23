"""
main.py — 语义搜索命令行入口

运行方式：
    python main.py

首次运行会自动构建索引（下载模型约 80MB + 编码文档）。
后续运行直接加载已有索引，启动更快。
"""

from __future__ import annotations

import os

from rich.console import Console
from rich.table import Table
from rich.text import Text

from data import DOCUMENTS
from indexer import CHROMA_PATH, build_index, load_index
from searcher import search

console = Console()


def format_score(score: float) -> Text:
    """根据分数高低返回带颜色的 Text 对象。"""
    score_str = f"{score:.4f}"
    if score >= 0.6:
        return Text(score_str, style="bold green")
    elif score >= 0.4:
        return Text(score_str, style="bold yellow")
    else:
        return Text(score_str, style="red")


def print_results(results: list[dict]) -> None:
    """用 Rich Table 美化输出搜索结果。"""
    if not results:
        console.print("[yellow]未找到相关文档[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("排名", style="dim", width=4, justify="center")
    table.add_column("分数", width=8, justify="center")
    table.add_column("文档内容", min_width=50, max_width=80)

    for rank, result in enumerate(results, start=1):
        # 文档内容超长时截断显示
        doc_text = result["document"]
        if len(doc_text) > 120:
            doc_text = doc_text[:117] + "..."

        table.add_row(
            str(rank),
            format_score(result["score"]),
            doc_text,
        )

    console.print(table)


def main():
    # 判断是否需要构建索引
    if os.path.exists(CHROMA_PATH):
        console.print(f"[dim]加载已有索引：{CHROMA_PATH}[/dim]")
        collection, model = load_index()
        console.print(f"[green]索引加载完成，共 {collection.count()} 条文档[/green]")
    else:
        console.print("[bold]首次运行，构建索引中...[/bold]")
        collection, model = build_index(DOCUMENTS)
        console.print(f"[green]索引构建完成，共 {collection.count()} 条文档[/green]")

    console.print(f"[dim]模型：all-MiniLM-L6-v2[/dim]\n")
    console.print("[bold cyan]语义搜索引擎已就绪[/bold cyan]")
    console.print("[dim]输入查询后按 Enter 搜索，输入 q 退出[/dim]\n")

    # 交互式查询循环
    while True:
        try:
            query = console.input("[bold]查询> [/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]已退出[/dim]")
            break

        if not query:
            continue
        if query.lower() == "q":
            console.print("[dim]已退出[/dim]")
            break

        results = search(query, collection, model, top_k=5)
        console.print(f"\n[dim]查询：{query}[/dim]")
        print_results(results)
        console.print()


if __name__ == "__main__":
    main()
