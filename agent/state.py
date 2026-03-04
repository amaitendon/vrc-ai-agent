"""
agent/state.py

LangGraphエージェントのステート定義。
エージェントの判断に必要な全情報をここに集約する。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core.types import OscStatus, ToolCallRecord


class AgentState(TypedDict):
    """
    LangGraphエージェントのメインステート。

    設計方針:
    - エージェントの判断に使う情報はすべてここに集約する
    - Phase2で空間情報や音声方向を追加する場合もここに足すだけで済む構造
    """

    # ── 会話履歴 ──────────────────────────────────────────────────────────────
    # add_messages: 上書きではなく追記されるReducer（LangGraph組み込み）
    # trim_messagesはノード処理前に適用してトークンを管理する
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 外部コンテキスト（各ノード処理前にシステムプロンプトへ差し込む） ────
    current_time: datetime
    current_date: str  # 例: "2026-02-26 (Thu)"

    # ── OSC受信ステータス（Phase1: 速度・回転のみ） ──────────────────────────
    # 位置情報はOSCで取得不可のため Phase2（視覚MCP）へ
    osc_status: OscStatus

    # ── ツール使用履歴（ナッジノードが参照する） ────────────────────────────
    # Phase2のナッジ実装時に使用。Phase1から記録だけしておく。
    tool_call_history: list[ToolCallRecord]

    # ── 最終アクションのタイムスタンプ（ナッジ判定用） ──────────────────────
    last_spoke_at: datetime | None  # 最後にTTSで発声した時刻
    last_memory_saved_at: datetime | None  # 最後に長期記憶を保存した時刻

    # ── 記憶保存ナッジ用（familiar-ai） ────────────────────────────────────
    unsaved_cycles: int
    prev_was_end_action: bool
    day_summary_context: str  # 起動時に取得した過去数日分の要約テキスト
    # ── Phase2: 以下は将来追加予定 ───────────────────────────────────────────
    # visual_context: str | None        # 視覚MCP（Spout）の解析結果
    # speaker_direction: float | None   # 音声L/R比較による話者方向
    # heartbeat_count: int              # ハートビートの発火回数
