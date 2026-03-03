import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from agent.graph import route_after_think, inject_tool_error, TOOL_CALL_REQUIRED_MESSAGE


@pytest.mark.asyncio
async def test_inject_tool_error_node():
    state = {"messages": []}
    result = await inject_tool_error(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)
    assert result["messages"][0].content == TOOL_CALL_REQUIRED_MESSAGE


def test_route_after_think_retry_logic():
    # ケース1: ツール呼び出しなし -> リトライへ（初回）
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Plain text response"),
        ]
    }
    assert route_after_think(state) == "inject_tool_error"

    # ケース2: ツール呼び出しなし -> リトライ1回済み -> ENDへ
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Plain text response"),
            SystemMessage(content=TOOL_CALL_REQUIRED_MESSAGE),
            AIMessage(content="Still plain text response"),
        ]
    }
    assert route_after_think(state) == END

    # ケース3: 別のターン（新たなHumanMessage以降）ならリトライ可能
    state = {
        "messages": [
            HumanMessage(content="Old turn"),
            AIMessage(content="Old response"),
            SystemMessage(content=TOOL_CALL_REQUIRED_MESSAGE),
            AIMessage(content="Old retry"),
            HumanMessage(content="New turn (from user)"),
            AIMessage(content="New plain text response"),
        ]
    }
    # 直近のHumanMessage以降にはSystemMessageがないのでリトライされるべき
    assert route_after_think(state) == "inject_tool_error"


def test_route_after_think_with_tools():
    # ツール呼び出しがあれば正常に "tools" へ
    state = {
        "messages": [
            AIMessage(
                content="", tool_calls=[{"name": "some_tool", "args": {}, "id": "1"}]
            )
        ]
    }
    assert route_after_think(state) == "tools"
