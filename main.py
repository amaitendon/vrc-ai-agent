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
from dotenv import load_dotenv


from langchain_core.messages import AIMessage, HumanMessage
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


# ── trim_history ─────────────────────────────────────────────────────────────


def _strip_images(messages: list) -> list:
    """
    古いメッセージ（末尾のmax_keep件を除く先頭側）から画像データを除去する。
    メッセージ自体は残し、content リスト内の image_url エントリだけ削除する。
    テキストが残らなくなる場合は "[image removed]" を代わりに入れる。
    """
    result = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_parts = [p for p in msg.content if p.get("type") != "image_url"]
            if not text_parts:
                text_parts = [{"type": "text", "text": "[image removed]"}]
            msg = msg.model_copy(update={"content": text_parts})
        result.append(msg)
    return result


def _split_into_blocks(messages: list) -> list[list]:
    """
    メッセージリストをブロック単位に分割する。

    ブロック定義:
      - HumanMessage                         → 単独1件のブロック
      - AIMessage (tool_calls なし)          → 単独1件のブロック
      - AIMessage (tool_calls あり)
          + それに対応する ToolMessage 群    → まとめて1ブロック

    ToolCall/ToolMessage ペアを分断しないことで LiteLLM の
    "Missing corresponding tool call" エラーを防ぐ。
    """
    blocks = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # tool_call_id の集合を取得
            expected_ids = {tc["id"] for tc in msg.tool_calls}
            block = [msg]
            i += 1
            # 対応する ToolMessage をすべて回収
            while i < len(messages) and expected_ids:
                next_msg = messages[i]
                if (
                    hasattr(next_msg, "tool_call_id")
                    and next_msg.tool_call_id in expected_ids
                ):
                    block.append(next_msg)
                    expected_ids.discard(next_msg.tool_call_id)
                    i += 1
                else:
                    break
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1
    return blocks


def trim_history(messages: list, max_messages: int) -> list:
    """
    会話履歴を max_messages 件以内に収める。

    Phase 1: 画像除去
      先頭ブロックから順に画像を除去してメッセージ数を変えずに軽量化を試みる。
      （現状は全メッセージの画像を除去する簡易実装）

    Phase 2: ブロック単位でのメッセージ削除
      先頭ブロックから削除して max_messages 以内に収める。
      ToolCall + ToolMessage のペアは必ず一緒に削除する。
    """
    if len(messages) <= max_messages:
        return messages

    # Phase 1: 画像除去
    messages = _strip_images(messages)
    if len(messages) <= max_messages:
        return messages

    # Phase 2: 先頭ブロックから削除
    blocks = _split_into_blocks(messages)
    while blocks and sum(len(b) for b in blocks) > max_messages:
        blocks.pop(0)

    return [msg for block in blocks for msg in block]


# ── queue_loop ────────────────────────────────────────────────────────────────


async def queue_loop(ctx: AppContext) -> None:
    """
    キューからイベントを取り出してグラフを1サイクルinvokeするメインループ。
    """
    graph = build_graph(priority_queue=ctx.priority_queue)
    logger.info("[queue_loop] started")

    from agent.nodes.action import memory_store
    from agent.memory_utils import backfill_day_summaries

    # バックグラウンドで過去の未生成Day Summaryを作成
    asyncio.create_task(backfill_day_summaries())

    summaries = await memory_store.recall_day_summaries_async(n=5)
    day_summary_context = memory_store.format_day_summaries_for_context(summaries)

    # グラフの初期ステートのベース（セッション全体で引き継ぐ情報）
    persistent_state: dict = {
        "messages": [],
        "tool_call_history": [],
        "last_spoke_at": None,
        "last_memory_saved_at": None,
        "unsaved_cycles": 0,
        "prev_was_end_action": False,
        "day_summary_context": day_summary_context,
    }

    while True:
        event: QueueEvent = await ctx.priority_queue.get()
        logger.info(
            f"[queue_loop] processing event: priority={event.priority} text={event.text!r}"
        )

        # interrupted_action があれば会話履歴に付与
        if event.interrupted_action:
            prefix = f"([Auto System Message] User spoke over the `{event.interrupted_action}` playback.)"
            user_message = HumanMessage(content=f"{prefix}{event.text}")
        else:
            user_message = HumanMessage(content=event.text)

        # 割り込み考慮後メッセージを追加
        persistent_state["messages"].append(user_message)

        # メッセージ履歴のトリム（ブロック単位 + 画像優先削除）
        max_history = int(os.environ.get("MAX_HISTORY", "30"))
        pre_count = len(persistent_state["messages"])
        persistent_state["messages"] = trim_history(
            persistent_state["messages"], max_messages=max_history
        )
        post_count = len(persistent_state["messages"])
        if pre_count != post_count:
            logger.debug(
                f"[queue_loop] trim_history: {pre_count} → {post_count} messages"
            )

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
            persistent_state["unsaved_cycles"] = result.get(
                "unsaved_cycles", persistent_state["unsaved_cycles"]
            )
            # BUG-1修正: サイクルが終了したら prev_was_end_action をリセットする
            # こうすることで、次のサイクルの最初のend_actionで再びナッジが発火するようになる
            persistent_state["prev_was_end_action"] = False

        except Exception as e:
            logger.error(f"[queue_loop] graph invocation error: {e}")

            # タイムアウトや一時的なネットワークエラーの場合のみリカバリ（再試行）を行う
            err_str = str(e).lower()
            is_transient = (
                isinstance(e, (asyncio.TimeoutError, TimeoutError))
                or "timeout" in err_str
            )

            if is_transient:
                # キューに溜まった残りのイベントをすべてドレインしてメッセージ履歴へ追加
                retry_needed = False
                while not ctx.priority_queue.empty():
                    try:
                        extra_event = ctx.priority_queue.get_nowait()

                        if extra_event.interrupted_action:
                            prefix = f"（{extra_event.interrupted_action}の再生中に割り込みがありました）"
                            extra_msg = HumanMessage(
                                content=f"{prefix}{extra_event.text}"
                            )
                        else:
                            extra_msg = HumanMessage(content=extra_event.text)

                        persistent_state["messages"].append(extra_msg)
                        ctx.priority_queue.task_done()
                        retry_needed = True
                    except asyncio.QueueEmpty:
                        break

                if retry_needed:
                    logger.info(
                        "[queue_loop] Retrying graph invocation with drained events after error."
                    )
                    try:
                        # まとめてAIを再呼び出し
                        retry_state: AgentState = {
                            **persistent_state,
                            "current_time": datetime.now(),
                            "current_date": datetime.now().strftime("%Y-%m-%d (%a)"),
                            "osc_status": ctx.osc_status,
                        }
                        result = await graph.ainvoke(retry_state)
                        persistent_state["messages"] = result["messages"]
                        persistent_state["tool_call_history"] = result.get(
                            "tool_call_history", []
                        )
                        persistent_state["last_spoke_at"] = result.get("last_spoke_at")
                        persistent_state["last_memory_saved_at"] = result.get(
                            "last_memory_saved_at"
                        )
                        persistent_state["unsaved_cycles"] = result.get(
                            "unsaved_cycles", persistent_state["unsaved_cycles"]
                        )
                        # BUG-1修正: リトライ成功後も prev_was_end_action をリセットする（正常系と同じ挙動）
                        persistent_state["prev_was_end_action"] = False
                    except Exception as e2:
                        logger.error(
                            f"[queue_loop] retry after timeout also failed: {e2}"
                        )

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
        filter=lambda record: "chat" not in record["extra"],
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

    load_dotenv()
    asyncio.run(main())

