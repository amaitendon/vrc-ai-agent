"""
agent/graph.py

LangGraphエージェントのグラフ定義。
ReActループの骨格: inject_context → think → ToolNode → (ループ or END)
"""

from __future__ import annotations

import asyncio

from datetime import datetime

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from agent.nodes.action import TOOLS
from agent.nodes.inject_context import inject_context
from agent.nodes.think import think
from agent.state import AgentState

_tool_node = ToolNode(TOOLS)

# ── リトライ設定 ──────────────────────────────────────────────────────────────

MAX_TOOL_RETRY = 1

TOOL_CALL_REQUIRED_MESSAGE = (
    "ERROR: Your previous response was rejected. "
    "You MUST respond by calling a tool. "
    "Plain text responses and code blocks (e.g. tool_code) are not allowed. "
    "Use the structured tool-calling interface provided by the API."
)


async def inject_tool_error(state: AgentState):
    """ツール呼び出し失敗時のエラーメッセージを注入するノード"""
    return {"messages": [SystemMessage(content=TOOL_CALL_REQUIRED_MESSAGE)]}


async def tool_node_with_timestamp(state):
    """
    ツール実行ノードのラッパー。
    実行結果（ToolMessage）のコンテンツの先頭に実行時刻のタイムスタンプを付与する。
    また、記憶関連ツールからの特定の戻り値を検知してステートを更新する。
    """
    result = await _tool_node.ainvoke(state)
    now = datetime.now().strftime("%H:%M:%S")
    timestamped = []

    # マージ用のステート更新辞書
    state_updates = {"messages": timestamped}

    for msg in result.get("messages", []):
        content = msg.content
        if isinstance(content, str):
            # end_actionツール結果の確認
            if content.startswith("NUDGE: "):
                content = content[7:]  # end_actionの結果確認のための文字列であり、AIには見せないため、"NUDGE: " を取り除く
                state_updates["prev_was_end_action"] = True
            elif "action_ended" in content:
                state_updates["prev_was_end_action"] = False
            # Rememberツール結果の確認
            elif content.startswith("Remembered"):
                state_updates["unsaved_cycles"] = 0

            # 直接書き換えを避け、model_copyを使用
            msg = msg.model_copy(update={"content": f"[{now}] {content}"})

        logger.bind(chat=True).info(msg.model_dump_json())
        timestamped.append(msg)

    return state_updates


# ── 条件分岐 ─────────────────────────────────────────────────────────────────


def route_after_think(state: AgentState) -> str:
    """
    thinkノードの後の分岐。
    - ツール呼び出しあり → "tools"
    - ツール呼び出しなし → リトライ or END
    """
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        logger.warning("[route] last message is not AIMessage, routing to END")
        return END

    if not last_message.tool_calls:
        # ツールを呼ばずにテキストだけ返した場合はリトライ判定へ
        # 直近のHumanMessage以降のSystemMessage(リトライ用)の数を数える
        retry_count = 0
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                break
            if isinstance(m, SystemMessage) and "was rejected" in m.content:
                retry_count += 1

        if retry_count < MAX_TOOL_RETRY:
            logger.warning(
                f"[route] no tool calls → injecting error and retrying (retry_count={retry_count})"
            )
            return "inject_tool_error"

        logger.debug(f"[route] no tool calls and retry exhausted ({retry_count}) → END")
        return END

    # end_actionを含む場合もToolNodeへ通す（実行結果を会話履歴に残すため）
    tool_names = [tc["name"] for tc in last_message.tool_calls]
    logger.debug(f"[route] routing to tools: {tool_names}")
    return "tools"


# ── グラフ構築 ────────────────────────────────────────────────────────────────


def build_graph(priority_queue: asyncio.PriorityQueue) -> StateGraph:
    """
    エージェントグラフを構築して返す。
    priority_queueをクロージャで束縛してroute_after_toolsに渡す。

    ループ構造:
      inject_context → think → tools → [inject_context → think → ...] → END
    """

    def route_after_tools(state: AgentState) -> str:
        """
        ToolNodeের後の分岐。
        - end_actionの実行結果が含まれる → END
        - 高優先度キューにイベントあり → END（main_loopに処理を戻す）
        - それ以外 → inject_contextへループ
        """
        last_message = state["messages"][-1]

        # end_actionが実行されていたらEND
        if (
            isinstance(last_message, ToolMessage)
            and "action_ended" in last_message.content
        ):
            logger.debug("[route_after_tools] end_action executed → END")
            return END

        # 高優先度キューにイベントがあればEND（割り込み処理をmain_loopに委譲）
        if not priority_queue.empty():
            logger.debug("[route_after_tools] high-priority event in queue → END")
            return END

        logger.debug("[route_after_tools] → inject_context")
        return "inject_context"

    builder = StateGraph(AgentState)

    # ノード登録
    builder.add_node("inject_context", inject_context)
    builder.add_node("think", think)
    builder.add_node("tools", tool_node_with_timestamp)
    builder.add_node("inject_tool_error", inject_tool_error)

    # エントリーポイント
    builder.set_entry_point("inject_context")

    # エッジ定義
    builder.add_edge("inject_context", "think")
    builder.add_edge("inject_tool_error", "inject_context")
    builder.add_conditional_edges("think", route_after_think)
    builder.add_conditional_edges("tools", route_after_tools)

    return builder.compile()
