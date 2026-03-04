from datetime import datetime
from loguru import logger

from agent.state import AgentState


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

    unsaved_cycles = state.get("unsaved_cycles", 0) + 1

    return {
        "current_time": now,
        "current_date": now.strftime("%Y-%m-%d (%a)"),
        "unsaved_cycles": unsaved_cycles,
        # "osc_status": osc_status,  # Phase1で有効化
    }
