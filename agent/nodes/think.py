from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from loguru import logger

from agent.nodes.action import TOOLS
from agent.state import AgentState
from prompts.prompts import BASE_SYSTEM_PROMPT


async def think(state: AgentState) -> dict:
    """
    LLM思考ノード。
    ツールを呼ぶか / end_actionを呼ぶか / （将来）ハートビートに応じた行動を決める。
    思考内容の素テキストはVRCへ渡さない（ToolNode経由でのみ外部に出る）。
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    from agent.llm import get_llm  # 循環import回避のためここで import # noqa: PLC0415
    from core.context import AppContext  # noqa: PLC0415

    llm = get_llm().bind_tools(TOOLS)

    system_msg = SystemMessage(content=BASE_SYSTEM_PROMPT)

    # trim_messagesはPhase1（STEP3）で実装後、ここで適用する
    # messages = trim_messages(state["messages"], ...)
    messages = state["messages"]


    logger.debug(f"[think] invoking LLM, message_count={len(messages)}")

    # 最新の入力メッセージ(差分)をチャットログへ出力
    if not isinstance(messages[-1], ToolMessage):
        last_msg = messages[-1]
        logger.bind(chat=True).info(last_msg.model_dump_json())

    response: AIMessage = await llm.ainvoke([system_msg] + messages)

    # レスポンス内容(思考＋ツールコールの差分)をチャットログへ出力
    logger.bind(chat=True).info(response.model_dump_json())

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
