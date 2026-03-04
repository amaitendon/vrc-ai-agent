import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from agent.graph import inject_context, think, route_after_think


@pytest.mark.asyncio
async def test_inject_context():
    state = {"messages": []}
    result = await inject_context(state)

    assert "current_time" in result
    assert "current_date" in result
    assert isinstance(result["current_time"], datetime)
    assert isinstance(result["current_date"], str)


@pytest.mark.asyncio
async def test_think_node_updates_history(clean_app_context):
    # LLMのモック
    mock_llm = MagicMock()
    mock_response = AIMessage(
        content="Thinking...",
        tool_calls=[{"name": "end_action", "args": {}, "id": "call_1"}],
    )
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    state = {
        "messages": [HumanMessage(content="Hello")],
        "tool_call_history": [],
        "current_time": datetime.now(),
        "current_date": "2026-02-27",
    }

    with patch("agent.llm.get_llm", return_value=mock_llm):
        # think内部で _build_system_prompt が呼ばれ、そこで AppContext.get() される
        result = await think(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        assert len(result["tool_call_history"]) == 1
        assert result["tool_call_history"][0]["tool_name"] == "end_action"


def test_route_after_think_tools():
    state = {
        "messages": [
            AIMessage(
                content="", tool_calls=[{"name": "some_tool", "args": {}, "id": "1"}]
            )
        ]
    }
    assert route_after_think(state) == "tools"


def test_route_after_think_end():
    # ツール呼び出しがない場合（1回目の失敗）はリトライへ遷移する
    state = {"messages": [AIMessage(content="Hello")]}
    assert route_after_think(state) == "inject_tool_error"


@pytest.mark.asyncio
async def test_route_after_tools_end_action():
    import asyncio

    pq = asyncio.PriorityQueue()
    # build_graph内で定義されているクロージャを取得するのは難しいため、
    # 実際のグラフを構築して挙動を確認するか、ロジックを分離してテストしやすくするのが望ましいが、
    # 現状はbuild_graph経由で構築されたグラフの挙動を間接的にテストする。

    from agent.graph import build_graph

    _ = build_graph(pq)

    # ツールノードの結果が "action_ended" の場合、ENDに遷移することを確認したい
    # ここでは route_after_tools のロジックを直接テストする（内部関数なので少し工夫が必要）
    # リファクタリングして route_after_tools を外部に出すのも手だが、まずは現状で。

    # 内部関数にアクセスできないので、graph.invoke で確認
    # ただしLLMなども絡むので、ノード単体テストが優先。
    pass
