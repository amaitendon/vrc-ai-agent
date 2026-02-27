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
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage
from loguru import logger

from agent.graph import build_graph
from agent.state import AgentState, OscStatus, Vector3

# ── AppContext ────────────────────────────────────────────────────────────────


@dataclass
class AppContext:
    """
    プロセス全体で共有するミュータブルな状態。
    グラフ外のコンポーネント（audio_listener・sayツール）が参照・更新する。
    """

    priority_queue: asyncio.PriorityQueue = field(default_factory=asyncio.PriorityQueue)
    say_task: asyncio.Task | None = None
    osc_status: OscStatus = field(
        default_factory=lambda: OscStatus(
            velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_y=0.0,
        )
    )

    # シングルトン
    _instance: AppContext | None = None

    @classmethod
    def get(cls) -> AppContext:
        if cls._instance is None:
            cls._instance = AppContext()
        return cls._instance


# ── イベント型 ────────────────────────────────────────────────────────────────


@dataclass(order=True)
class QueueEvent:
    """
    優先度付きキューのエントリ。
    priority が小さいほど先に処理される（0が最高優先度）。
    """

    priority: int
    text: str = field(compare=False)
    interrupted_action: str | None = field(
        default=None, compare=False
    )  # 割り込み時のツール名


# 優先度定数
PRIORITY_VOICE = 0  # 音声入力（最高優先度）
PRIORITY_BEAT = 10  # ハートビート（Phase2）


# ── audio_listener ────────────────────────────────────────────────────────────


async def audio_listener(ctx: AppContext) -> None:
    """
    音声を常時監視するタスク。このタスクは絶対に止まらない。

    音声検知時:
      - sayが再生中なら再生をキャンセルして interrupted_action を付与
      - キューに高優先度でput
    """
    # TODO Phase1（STEP3）: aiavatarkitのVAD+STTに差し替える
    # 現状はダミー実装（標準入力でテキスト入力をエミュレート）
    logger.info("[audio_listener] started (dummy stdin mode)")

    loop = asyncio.get_event_loop()
    while True:
        try:
            # ダミー: 標準入力から読み取り（STT実装後に差し替え）
            text = await loop.run_in_executor(None, input, "user> ")
            if not text.strip():
                continue

            interrupted_action = None

            # sayが再生中なら停止
            if ctx.say_task and not ctx.say_task.done():
                logger.info(
                    "[audio_listener] cancelling say_task due to voice interrupt"
                )
                ctx.say_task.cancel()
                interrupted_action = "say"

            event = QueueEvent(
                priority=PRIORITY_VOICE,
                text=text.strip(),
                interrupted_action=interrupted_action,
            )
            await ctx.priority_queue.put(event)
            logger.debug(f"[audio_listener] queued: {event}")

        except Exception as e:
            # audio_listenerは何があっても止めない
            logger.error(f"[audio_listener] error (continuing): {e}")


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

        # 1サイクル分のステートを組み立て
        invoke_state: AgentState = {
            **persistent_state,
            "messages": persistent_state["messages"] + [user_message],
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
