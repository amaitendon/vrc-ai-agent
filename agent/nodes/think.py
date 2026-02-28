from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage
from loguru import logger

from agent.nodes.action import TOOLS
from agent.state import AgentState


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
            "再生が完了するまで say を呼び出さないでください。"
        )

    context = "\n".join(context_lines)
    return f"{BASE_SYSTEM_PROMPT}\n\n--- 現在のコンテキスト ---\n{context}"


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
    
    # 最新の入力メッセージ(差分)をチャットログへ出力
    if messages:
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
