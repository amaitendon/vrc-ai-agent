from langchain_core.tools import tool

from actuators.speech import say

# Phase1で追加予定:
#   from actuators.chat_box import chat
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
    say,
    # chat,
    # move,
    # save_memory,
]
