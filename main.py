"""
main.py

エントリーポイント。
以下の3つの非同期タスクを起動して並行実行する。

  1. audio_listener  : 音声を常時監視。検知したら即キューにput（絶対に止まらない）
  2. queue_loop      : キューからイベントを取り出してグラフを1サイクルinvoke
  3. osc_receiver    : OSCステータスを常時受信してAppContextを更新（Phase1後半）
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage, trim_messages
from loguru import logger

from agent.graph import build_graph
from agent.state import AgentState
from inputs.audio import setup_audio_listener
from core.context import AppContext, QueueEvent


# ── audio_listener ────────────────────────────────────────────────────────────


async def audio_listener(ctx: AppContext) -> None:
    """
    音声を常時監視するタスク。このタスクは絶対に止まらない。
    VAD + FasterWhisper を使って STT を行い、結果をキューに投げる。
    """
    logger.info("[audio_listener] starting real STT pipeline")

    try:
        await setup_audio_listener(ctx)
    except Exception as e:
        logger.error(f"[audio_listener] fatal error: {e}", exc_info=True)


# ── queue_loop ────────────────────────────────────────────────────────────────


async def queue_loop(ctx: AppContext) -> None:
    """
    キューからイベントを取り出してグラフを1サイクルinvokeするメインループ。
    """
    graph = build_graph(priority_queue=ctx.priority_queue)
    logger.info("[queue_loop] started")

    # グラフの初期ステートのベース（セッション全体で引き継ぐ情報）
    persistent_state: dict = {
        "messages": [],
        "tool_call_history": [],
        "last_spoke_at": None,
        "last_memory_saved_at": None,
    }

    while True:
        event: QueueEvent = await ctx.priority_queue.get()
        logger.info(
            f"[queue_loop] processing event: priority={event.priority} text={event.text!r}"
        )

        # interrupted_action があれば会話履歴に付与
        if event.interrupted_action:
            prefix = f"（{event.interrupted_action}の再生中に割り込みがありました）"
            user_message = HumanMessage(content=f"{prefix}{event.text}")
        else:
            user_message = HumanMessage(content=event.text)

        # 割り込み考慮後メッセージを追加
        persistent_state["messages"].append(user_message)

        # トークン（メッセージ数）の上限管理 (最新10往復=約20メッセージ程度を残す)
        # trim_messagesを用いて、システムプロンプト等を含まない生の履歴リストを切り詰める
        try:
            trimmed_messages = trim_messages(
                persistent_state["messages"],
                max_tokens=20,  # 簡易的にメッセージ件数として扱う（token_counter未指定時のデフォルト挙動）
                strategy="last",
                token_counter=len,  # len関数でシンプルにリストの長さとして計算
                include_system=False,  # ここにはSystemMessageは含まれていない
                allow_partial=False,
            )
            persistent_state["messages"] = trimmed_messages
        except Exception as e:
            logger.warning(f"[queue_loop] trim_messages error: {e}")

        # 1サイクル分のステートを組み立て
        invoke_state: AgentState = {
            **persistent_state,
            "current_time": datetime.now(),
            "current_date": datetime.now().strftime("%Y-%m-%d (%a)"),
            "osc_status": ctx.osc_status,
        }

        try:
            result = await graph.ainvoke(invoke_state)
            # 会話履歴とツール使用履歴をセッション間で引き継ぐ
            persistent_state["messages"] = result["messages"]
            persistent_state["tool_call_history"] = result.get("tool_call_history", [])
            persistent_state["last_spoke_at"] = result.get("last_spoke_at")
            persistent_state["last_memory_saved_at"] = result.get(
                "last_memory_saved_at"
            )

        except Exception as e:
            logger.error(f"[queue_loop] graph invocation error: {e}")

        finally:
            ctx.priority_queue.task_done()


# ── エントリーポイント ────────────────────────────────────────────────────────


async def main() -> None:
    logger.info("VRC AI Agent starting...")
    ctx = AppContext.get()

    await asyncio.gather(
        audio_listener(ctx),
        queue_loop(ctx),
        # osc_receiver(ctx),  # Phase1後半で有効化
    )


if __name__ == "__main__":
    asyncio.run(main())
