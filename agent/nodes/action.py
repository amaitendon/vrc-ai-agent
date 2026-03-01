from langchain_core.tools import tool

from actuators.speech import say
from actuators.chat_box import chat_box
from actuators.movement import move, look_direction, jump
from inputs.vision import get_current_view
# from memory.long_term import save_memory


@tool
def end_action() -> str:
    """
    現在の行動サイクルを終了する。
    ユーザーからの応答を待つ、またはこれ以上行動が不要と判断したときに呼ぶ。
    """
    return "action_ended"


# ToolNodeに登録するツール一覧
# ツールを追加するときはこのリストに足すだけでグラフ構造は変わらない
TOOLS = [
    end_action,
    say,
    chat_box,
    move,
    look_direction,
    jump,
    get_current_view,
    # save_memory,
]
