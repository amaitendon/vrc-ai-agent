from langchain_core.tools import tool

from actuators.speech import say
from actuators.chat_box import chat_box
from actuators.movement import move, rotate, jump
from inputs.vision import get_current_view
# from memory.long_term import save_memory


@tool
def end_action() -> str:
    """
    End the current action cycle.
    Call this when you are waiting for a user response, or when no further actions are needed.
    """
    return "action_ended"


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
    # save_memory,
]
