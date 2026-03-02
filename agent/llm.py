from __future__ import annotations

import os
from typing import Optional

from langchain_litellm import ChatLiteLLM
from loguru import logger

# シングルトン用のグローバル変数
_llm_instance: Optional[ChatLiteLLM] = None
_current_model: Optional[str] = None

# トークナイザーキャッシュ
_local_tokenizer = None
_tokenizer_model: Optional[str] = None


def get_llm() -> ChatLiteLLM:
    """
    環境変数 LITELLM_MODEL に基づいてLLMインスタンスを返す（シングルトン）。
    モデルが変更された場合のみ再生成する。
    """
    global _llm_instance, _current_model
    model = os.environ.get("LITELLM_MODEL", "gemini/gemini-2.0-flash")

    if _llm_instance is None or _current_model != model:
        logger.debug(f"[llm] initializing new ChatLiteLLM instance: {model}")
        _llm_instance = ChatLiteLLM(
            model=model,
            temperature=0.7,
        )
        _current_model = model

    return _llm_instance


def get_local_tokenizer():
    """
    google-genai の LocalTokenizer を用いて Gemini のローカルトークナイザーを取得する。
    初回のみモデルのダウンロードが発生するため、数秒かかる場合があります。
    """
    global _local_tokenizer, _tokenizer_model
    model_name = os.environ.get("LITELLM_MODEL", "gemini/gemini-2.0-flash")
    # LiteLLM形式 (provider/model) から model名のみ抽出
    model_id = model_name.split("/")[-1]

    if _local_tokenizer is None or _tokenizer_model != model_id:
        if "gemini" in model_name:
            try:
                from google.genai.local_tokenizer import LocalTokenizer

                logger.debug(f"[llm] loading local tokenizer for {model_id}...")
                # 直接インスタンス化 (model_name="gemini-2.0-flash" など)
                _local_tokenizer = LocalTokenizer(model_name=model_id)
                _tokenizer_model = model_id
                logger.debug(f"[llm] local tokenizer for {model_id} loaded.")
            except Exception as e:
                logger.warning(f"[llm] failed to load local tokenizer: {e}")
                return None
        else:
            return None

    return _local_tokenizer


def count_tokens_locally(messages: list) -> int:
    """
    ローカルトークナイザーを使用してメッセージリストの合計トークン数を計算する。
    Gemini以外、またはエラー時は ChatLiteLLM の標準メソッド（遅い）にフォールバックする。
    """
    tokenizer = get_local_tokenizer()
    if tokenizer:
        try:
            total_tokens = 0
            for msg in messages:
                # LocalTokenizer.count_tokens() は total_tokens 属性を持つオブジェクトを返す
                res = tokenizer.count_tokens(msg.content)
                total_tokens += res.total_tokens
            return total_tokens
        except Exception as e:
            logger.warning(f"[llm] local token count failed, fallback: {e}")

    # フォールバック
    return get_llm().get_num_tokens_from_messages(messages)
