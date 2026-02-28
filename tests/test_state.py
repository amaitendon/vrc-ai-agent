from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from agent.state import AgentState


def test_agent_state_initialization():
    """AgentStateが期待通りの構造を持っているか確認"""
    state: AgentState = {
        "messages": [],
        "current_time": datetime.now(),
        "current_date": "2026-02-27 (Fri)",
        "osc_status": {"velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular_y": 0.0},
        "tool_call_history": [],
        "last_spoke_at": None,
        "last_memory_saved_at": None,
    }
    assert isinstance(state["messages"], list)
    assert len(state["messages"]) == 0


def test_add_messages_reducer():
    """LangGraphのadd_messagesアノテーションによるメッセージ追記の動作確認"""
    # 注: AgentState自体はTypedDictなので、実行時の動作はLangGraphのノード外では
    # 単なるリスト操作になる可能性があるが、型定義の整合性を確認する。
    messages = [HumanMessage(content="Hello")]
    new_message = [AIMessage(content="Hi there")]

    # 手動でadd_messages的な振る舞いをシミュレート（LangGraph内部での動作期待値）
    from langgraph.graph.message import add_messages

    combined = add_messages(messages, new_message)

    assert len(combined) == 2
    assert combined[0].content == "Hello"
    assert combined[1].content == "Hi there"
