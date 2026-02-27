"""
core/types.py

プロセス全体で共有する汎用的なデータ型。
"""

from __future__ import annotations

from datetime import datetime
from typing_extensions import TypedDict


class Vector3(TypedDict):
    """3次元ベクトル。VRCのOSC受信値などに使用。"""

    x: float
    y: float
    z: float


class OscStatus(TypedDict):
    """OSC経由で取得できる自分自身の状態。"""

    velocity: Vector3  # 移動速度ベクトル（移動中かどうかの確認用）
    angular_y: float  # Y軸回転速度


class ToolCallRecord(TypedDict):
    """ナッジノード用のツール使用履歴エントリ。"""

    tool_name: str
    called_at: datetime
