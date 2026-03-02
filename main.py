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
import os

from langchain_core.messages import HumanMessage, trim_messages
from loguru import logger

from agent.graph import build_graph
from agent.state import AgentState
from agent.llm import get_llm, count_tokens_locally
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

        # トークン（メッセージ数）の上限管理
        # trim_messagesを用いて、設定されたトークン数に収まるように履歴リストを切り詰める
        try:
            max_history = int(os.environ.get("MAX_HISTORY", 40))

            # ローカル計算を用いたトークン数計算
            pre_tokens = len(persistent_state["messages"])
            logger.debug(
                f"[queue_loop] messages before trim: {len(persistent_state['messages'])}, tokens: {pre_tokens}"
            )

            trimmed_messages = trim_messages(
                persistent_state["messages"],
                max_tokens=max_history,
                strategy="last",
                token_counter=len, # count_tokens_locally,：msg.contentがリストや辞書である場合、LocalTokenizerが処理可能な形式に変換が必要
                include_system=False,  # ここにはSystemMessageは含まれていない
                allow_partial=False,
            )
            persistent_state["messages"] = trimmed_messages

            # トリム後のトークン数をログ出力
            post_tokens = len(persistent_state["messages"])
            logger.debug(
                f"[queue_loop] messages after trim: {len(persistent_state['messages'])}, tokens: {post_tokens}"
            )

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
    import sys
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # デフォルトの標準出力ハンドラの設定変更
    logger.remove()
    logger.add(sys.stderr, filter=lambda record: "chat" not in record["extra"])

    # 通常のログファイルの設定 (例: 2026-02-28-1309.log)
    log_filename = datetime.now().strftime("%Y-%m-%d-%H%M.log")
    logger.add(
        log_dir / log_filename,
        # filter=lambda record: "chat" not in record["extra"],
        enqueue=True,
    )

    # チャット履歴専用ログファイルの設定 (例: 2026-02-28-1309-chat.log)
    chat_log_filename = datetime.now().strftime("%Y-%m-%d-%H%M-chat.log")
    logger.add(
        log_dir / chat_log_filename,
        filter=lambda record: "chat" in record["extra"],
        format="{message}",
        enqueue=True,
    )

    asyncio.run(main())
