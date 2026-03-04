"""
manual_test_day_summary.py

day_summary_context の内容をDBから直接取得して確認するスクリプト。
エージェントを起動せずに実行できる。

使い方:
    uv run python tests/manual_test_day_summary.py
"""

import asyncio
from agent.nodes.action import memory_store


async def main() -> None:
    summaries = await memory_store.recall_day_summaries_async(n=5)
    context = memory_store.format_day_summaries_for_context(summaries)

    if not context:
        print("=== day_summary_context: (empty — no summaries in DB yet) ===")
    else:
        print("=== day_summary_context ===")
        print(context)
        print("===========================")

    print(f"\n({len(summaries)} day summaries found)")


if __name__ == "__main__":
    asyncio.run(main())
