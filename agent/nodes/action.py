import os

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated
from pydantic import Field

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
# 補足1: nudge_pending時に呼ばれるとnudge_doneがセットされるが、現状この値はグラフ外（main.py）
#       で消去されるため、グラフ内でのみ意味を持つ一時的な状態となる（意図的）。
# 補足2: ここでAIがナッジを無視して再度end_actionを呼んだ場合、nudge_state == "nudge_pending" の
#       分岐に入りそのままサイクルが終了する。これも意図通り。
@tool
def end_action(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    End the current action cycle.
    Call this when you are waiting for a user response, or when no further actions are needed.
    If no other tool is applicable, you ARE REQUIRED to call this.
    """
    unsaved_cycles = state.get("unsaved_cycles", 0)
    nudge_state = state.get("nudge_remember", "idle")

    max_history = int(os.getenv("MAX_HISTORY", "30"))
    strong_nudge_threshold = max_history // 3
    weak_nudge_threshold = max_history // 4

    if nudge_state == "nudge_pending":
        msg = ToolMessage(content="action_ended", tool_call_id=tool_call_id, name="end_action")
        return Command(update={"nudge_remember": "nudge_done", "messages": [msg]})
    elif unsaved_cycles >= strong_nudge_threshold:
        msg = ToolMessage(content=f"IMPORTANT: {unsaved_cycles} cycles have passed without saving any memories. Memories will be lost — use `remember` to save important ones now.", tool_call_id=tool_call_id, name="end_action")
        return Command(update={"nudge_remember": "nudge_pending", "messages": [msg]})
    elif unsaved_cycles >= weak_nudge_threshold:
        msg = ToolMessage(content=f"{unsaved_cycles} cycles have passed without saving any memories. Consider using `remember` to save recent events.", tool_call_id=tool_call_id, name="end_action")
        return Command(update={"nudge_remember": "nudge_pending", "messages": [msg]})

    msg = ToolMessage(content="action_ended", tool_call_id=tool_call_id, name="end_action")
    return Command(update={"nudge_remember": "idle", "messages": [msg]})


@tool
async def remember(
    content: Annotated[str, Field(description="What to remember (1-3 sentences).")],
    *,
    emotion: Annotated[
        str,
        Field(
            description='Emotional tone of this memory. One of: "neutral", "happy", "sad", "curious", "excited", "moved".'
        ),
    ] = "neutral",
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Save something to long-term memory. Use this to remember important things: what you saw, what happened, how you felt, conversations.
    """
    ok = await memory_store.save_async(
        content=content, kind="observation", emotion=emotion, image_path=None
    )
    if ok:
        msg = ToolMessage(content=f"Remembered: {content[:60]}", tool_call_id=tool_call_id, name="remember")
        return Command(update={"unsaved_cycles": 0, "messages": [msg]})
    msg = ToolMessage(content="Failed to save memory.", tool_call_id=tool_call_id, name="remember")
    return Command(update={"messages": [msg]})


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
