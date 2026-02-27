"""
agent/graph.py

LangGraphエージェントのグラフ定義。
ReActループの骨格: inject_context → think → ToolNode → (ループ or END)
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from agent.state import AgentState

# ── ツール定義（各モジュールが実装後にここへインポートして追加する） ──────────
# Phase1で追加予定:
#   from actuators.speech import say
#   from actuators.chat import chat
#   from actuators.movement import move
#   from memory.long_term import save_memory


@tool
def end_action() -> str:
    """
    現在の行動サイクルを終了する。
    会話への応答が完了した、またはこれ以上行動が不要と判断したときに呼ぶ。
    """
    return "action_ended"


# ToolNodeに登録するツール一覧
# ツールを追加するときはこのリストに足すだけでグラフ構造は変わらない
TOOLS = [
    end_action,
    # say,
    # chat,
    # move,
    # save_memory,
]

tool_node = ToolNode(TOOLS)


# ── ノード定義 ────────────────────────────────────────────────────────────────


async def inject_context(state: AgentState) -> dict:
    """
    外部コンテキストをステートに注入するノード。
    時刻・OSCステータスなどを最新値に更新する。

    このノードはthinkノードの直前に毎回実行されるため、
    LLMは常に最新のコンテキストを参照できる。
    """
    now = datetime.now()
    logger.debug(f"[inject_context] time={now.isoformat()}")

    # TODO Phase1: OSCレシーバーから velocity / angular_y を取得して更新
    # osc_status = osc_receiver.get_latest()

    return {
        "current_time": now,
        "current_date": now.strftime("%Y-%m-%d (%a)"),
        # "osc_status": osc_status,  # Phase1で有効化
    }


async def think(state: AgentState) -> dict:
    """
    LLM思考ノード。
    ツールを呼ぶか / end_actionを呼ぶか / （将来）ハートビートに応じた行動を決める。
    思考内容の素テキストはVRCへ渡さない（ToolNode経由でのみ外部に出る）。
    """
    from agent.llm import get_llm  # 循環import回避のためここでimport

    llm = get_llm().bind_tools(TOOLS)

    # 外部コンテキストをシステムプロンプトへ動的に差し込む
    system_content = _build_system_prompt(state)
    system_msg = SystemMessage(content=system_content)

    # trim_messagesはPhase1（STEP3）で実装後、ここで適用する
    # messages = trim_messages(state["messages"], ...)
    messages = state["messages"]

    logger.debug(f"[think] invoking LLM, message_count={len(messages)}")
    response: AIMessage = await llm.ainvoke([system_msg] + messages)
    logger.debug(
        f"[think] response tool_calls={[tc['name'] for tc in response.tool_calls]}"
    )

    # ツール使用履歴を記録（Phase2のナッジが参照する）
    now = datetime.now()
    new_records = [
        {"tool_name": tc["name"], "called_at": now} for tc in response.tool_calls
    ]
    updated_history = state.get("tool_call_history", []) + new_records

    return {
        "messages": [response],
        "tool_call_history": updated_history,
    }


# ── 条件分岐 ─────────────────────────────────────────────────────────────────


def route_after_think(state: AgentState) -> str:
    """
    thinkノードの後の分岐。
    - ツール呼び出しあり → "tools"
    - end_actionが含まれる → END
    - ツール呼び出しなし（フォールバック） → END
    """
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        logger.warning("[route] last message is not AIMessage, routing to END")
        return END

    if not last_message.tool_calls:
        # ツールを呼ばずにテキストだけ返した場合はENDへ（フォールバック）
        logger.debug("[route] no tool calls → END")
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
        ToolNodeの後の分岐。
        - end_actionの実行結果が含まれる → END
        - 高優先度キューにイベントあり → END（main_loopに処理を戻す）
        - それ以外 → inject_contextへループ
        """
        last_message = state["messages"][-1]

        # end_actionが実行されていたらEND
        if (
            isinstance(last_message, ToolMessage)
            and last_message.content == "action_ended"
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
    builder.add_node("tools", tool_node)

    # エントリーポイント
    builder.set_entry_point("inject_context")

    # エッジ定義
    builder.add_edge("inject_context", "think")
    builder.add_conditional_edges("think", route_after_think)
    builder.add_conditional_edges("tools", route_after_tools)

    return builder.compile(
        # 無限ループ防止。1イベントあたりの最大ステップ数。
        # 1サイクル = inject_context + think + tools の3ステップ消費
        # → 25ステップ ≒ 最大8ツール呼び出し程度
    )


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _build_system_prompt(state: AgentState) -> str:
    """
    外部コンテキストを含むシステムプロンプトを動的に生成する。
    prompts.pyのベースプロンプトにコンテキストを付加する。
    """
    from core.context import AppContext  # noqa: PLC0415
    from prompts import BASE_SYSTEM_PROMPT  # noqa: PLC0415

    context_lines = [
        f"現在日時: {state.get('current_date', '')} {state.get('current_time', '')}",
    ]

    # Phase1: OSCステータスが取得できていれば追加
    osc = state.get("osc_status")
    if osc:
        v = osc.get("velocity", {})
        speed = (v.get("x", 0) ** 2 + v.get("y", 0) ** 2 + v.get("z", 0) ** 2) ** 0.5
        context_lines.append(f"移動速度: {speed:.2f} m/s")

    # sayが再生中であればLLMに通知
    # → 再生完了前にsayを重複呼び出し・end_actionを呼ぶのを防ぐ
    ctx = AppContext.get()
    if ctx.say_task and not ctx.say_task.done():
        context_lines.append(
            "【重要】現在あなたの音声が再生中です。"
            "再生が完了するまで say を呼び出さず、end_action も呼び出さないでください。"
        )

    context = "\n".join(context_lines)
    return f"{BASE_SYSTEM_PROMPT}\n\n--- 現在のコンテキスト ---\n{context}"
