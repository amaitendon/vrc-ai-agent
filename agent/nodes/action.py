import os

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from pydantic import Field
from pathlib import Path

from actuators.speech import say
from actuators.chat_box import chat_box
from actuators.movement import move, rotate, jump
from inputs.vision import get_current_view
from memory.memory import ObservationMemory

# Initialize memory store singleton
memory_store = ObservationMemory()


# [開発者向け]
# AIエージェントには「行動終了」ツールとして見えるが、サイクル終了前の
# 記憶保存ナッジ判定チェックも兼ねている。
#
# 動作フロー:
#   1回目の end_action 呼び出し
#     → unsaved_cycles が閾値超えなら "NUDGE: " プレフィックス付きで返す
#     → graph.py の tool_node_with_timestamp が "NUDGE: " を検知し
#       prev_was_end_action=True にセットして LLM に差し戻す
#   2回目の end_action 呼び出し（prev_was_end_action=True）
#     → ナッジをスキップして "action_ended" を返し、グラフを終了させる
#
# Note: "NUDGE: " プレフィックスによる文字列ベースのシグナルは技術的負債（WARN-1）。
#       将来は Command パターンへの置き換えを検討。
@tool
def end_action(state: Annotated[dict, InjectedState]) -> str:
    """
    End the current action cycle.
    Call this when you are waiting for a user response, or when no further actions are needed.
    If no other tool is applicable, you ARE REQUIRED to call this.
    """
    unsaved_cycles = state.get("unsaved_cycles", 0)
    prev_was_end_action = state.get("prev_was_end_action", False)

    max_history = int(os.getenv("MAX_HISTORY", "30"))
    strong_nudge_threshold = (
        max_history // 3
    )  # 3mesg/1cyc想定でコンテキストが溢れないように保存を促す
    weak_nudge_threshold = max_history // 4  # 安直に+1

    if not prev_was_end_action:
        if unsaved_cycles >= strong_nudge_threshold:
            return f"NUDGE: IMPORTANT: {unsaved_cycles} cycles have passed without saving any memories. Memories will be lost — use `remember` to save important ones now."
        elif unsaved_cycles >= weak_nudge_threshold:
            return f"NUDGE: {unsaved_cycles} cycles have passed without saving any memories. Consider using `remember` to save recent events."

    return "action_ended"


@tool
async def remember(
    content: Annotated[str, Field(description="What to remember (1-3 sentences).")],
    emotion: Annotated[
        str,
        Field(
            description='Emotional tone of this memory. One of: "neutral", "happy", "sad", "curious", "excited", "moved".'
        ),
    ] = "neutral",
    image_filename: Annotated[
        str | None,
        Field(
            description="Optional filename of the image returned by get_current_view()."
        ),
    ] = None,
) -> str:
    """
    Save something to long-term memory. Use this to remember important things: what you saw, what happened, how you felt, conversations.
    If you just took a photo, you can pass the image_filename to attach it.
    """
    image_path = None
    if image_filename:
        log_dir = Path(os.environ.get("VISION_LOG_DIR", "logs")).absolute()
        image_path = str(log_dir / image_filename)

    ok = await memory_store.save_async(
        content=content, kind="observation", emotion=emotion, image_path=image_path
    )
    if ok:
        suffix = " (with image)" if image_path else ""
        return f"Remembered{suffix}: {content[:60]}"
    return "Failed to save memory."


@tool
async def recall(
    query: Annotated[
        str, Field(description="Topic or keyword to search for in memory.")
    ],
    n: Annotated[
        int, Field(description="Number of memories to return (default 3).", ge=1, le=10)
    ] = 3,
) -> str:
    """
    Search long-term memory for things related to a topic.
    Use this to remember past observations, conversations, or feelings.
    """
    memories = await memory_store.recall_async(query, n=n)
    if not memories:
        return "No relevant memories found."
    lines = []
    for m in memories:
        score = f" ({m['score']:.2f})" if "score" in m else ""
        emotion = (
            f" [{m['emotion']}]" if m.get("emotion", "neutral") != "neutral" else ""
        )
        img = " 📷" if m.get("image_path") else ""
        lines.append(
            f"- {m['date']} {m['time']}{score}{emotion}{img}: {m['summary'][:120]}"
        )
    return "\n".join(lines)


# ToolNodeに登録するツール一覧
# ツールを追加するときはこのリストに足すだけでグラフ構造は変わらない
TOOLS = [
    end_action,
    say,
    chat_box,
    move,
    rotate,
    jump,
    get_current_view,
    remember,
    recall,
]
