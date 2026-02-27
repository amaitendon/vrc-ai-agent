"""
agent/llm.py

LiteLLM経由でLLMを取得するラッパー。
環境変数 LITELLM_MODEL を変更するだけでモデルを切り替えられる。

開発中の推奨設定（.env）:
  LITELLM_MODEL=gemini/gemini-2.5-flash-lite     # 無料枠が尽きるまで
  LITELLM_MODEL=openrouter/moonshotai/kimi-k2.5  # 無料枠終了後
"""

from __future__ import annotations

import os

from langchain_litellm import ChatLiteLLM
from loguru import logger


def get_llm() -> ChatLiteLLM:
    """
    環境変数 LITELLM_MODEL に基づいてLLMインスタンスを返す。
    呼び出しのたびに生成するためモデル切り替えが即時反映される。
    """
    model = os.environ.get("LITELLM_MODEL", "gemini/gemini-2.0-flash")
    logger.debug(f"[llm] using model: {model}")

    return ChatLiteLLM(
        model=model,
        temperature=0.7,
    )
