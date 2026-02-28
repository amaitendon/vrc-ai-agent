from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    trim_messages,
)


def test_trim_messages_logic():
    """
    main.py で行われている trim_messages の挙動確認テスト。
    """
    messages = []

    # 30件のメッセージを作成
    for i in range(30):
        if i % 2 == 0:
            messages.append(HumanMessage(content=f"User message {i}"))
        else:
            messages.append(AIMessage(content=f"AI message {i}"))

    assert len(messages) == 30

    # main.py と同じ切り詰め処理を実行
    trimmed_messages = trim_messages(
        messages,
        max_tokens=20,
        strategy="last",
        token_counter=len,
        include_system=False,
        allow_partial=False,
    )

    # 20件に切り詰められていることを確認
    assert len(trimmed_messages) == 20
    # 最新の20件（index 10～29）が残っていることを確認
    assert trimmed_messages[0].content == "User message 10"
    assert trimmed_messages[-1].content == "AI message 29"


def test_trim_messages_with_system():
    """
    SystemMessageが含まれる場合の挙動確認（将来の布石）
    """
    messages = [SystemMessage(content="You are a helpful AI.")]
    for i in range(25):
        messages.append(HumanMessage(content=f"Msg {i}"))

    trimmed = trim_messages(
        messages,
        max_tokens=20,
        strategy="last",
        token_counter=len,
        include_system=True,  # SystemMessageを残す指定
        allow_partial=False,
    )

    assert len(trimmed) == 20
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[-1].content == "Msg 24"
