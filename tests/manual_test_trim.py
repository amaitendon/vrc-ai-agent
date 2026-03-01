import asyncio
import os
from langchain_core.messages import HumanMessage, trim_messages
from loguru import logger

from agent.llm import get_llm


async def main():
    # テスト用に MAX_HISTORY_TOKENS を小さめに設定
    os.environ["MAX_HISTORY_TOKENS"] = "20"

    # 複数メッセージの作成。各メッセージが10トークン以上だとして、20トークンだと古いものが消えるはず
    messages = [
        HumanMessage(content="Hello, this is message 1. I am talking to you."),
        HumanMessage(content="Hello, this is message 2. How are you today?"),
        HumanMessage(
            content="Hello, this is message 3. This should definitely exceed 20 tokens."
        ),
    ]

    llm = get_llm()
    max_tokens = int(os.environ.get("MAX_HISTORY_TOKENS", 4000))

    pre_tokens = llm.get_num_tokens_from_messages(messages)
    logger.info(f"Messages before trim: {len(messages)}, tokens: {pre_tokens}")
    for i, m in enumerate(messages):
        t = llm.get_num_tokens_from_messages([m])
        logger.info(f"  [{i}] ({t} tokens): {m.content}")

    try:
        trimmed_messages = trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=llm.get_num_tokens_from_messages,
            include_system=False,
            allow_partial=False,
        )
        post_tokens = llm.get_num_tokens_from_messages(trimmed_messages)
        logger.info(
            f"\nMessages after trim: {len(trimmed_messages)}, tokens: {post_tokens}"
        )
        for i, m in enumerate(trimmed_messages):
            t = llm.get_num_tokens_from_messages([m])
            logger.info(f"  [{i}] ({t} tokens): {m.content}")

    except Exception as e:
        logger.error(f"trim_messages error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
