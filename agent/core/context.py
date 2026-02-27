"""
context.py

プロセス全体で共有するミュータブルな状態やイベントの定義。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agent.state import OscStatus, Vector3


# ── AppContext ────────────────────────────────────────────────────────────────


@dataclass
class AppContext:
    """
    プロセス全体で共有するミュータブルな状態。
    グラフ外のコンポーネント（audio_listener・sayツール）が参照・更新する。
    """

    priority_queue: asyncio.PriorityQueue = field(default_factory=asyncio.PriorityQueue)
    say_task: asyncio.Task | None = None
    osc_status: OscStatus = field(
        default_factory=lambda: OscStatus(
            velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_y=0.0,
        )
    )

    # シングルトン
    _instance: AppContext | None = None

    @classmethod
    def get(cls) -> AppContext:
        if cls._instance is None:
            cls._instance = AppContext()
        return cls._instance


# ── イベント型 ────────────────────────────────────────────────────────────────


@dataclass(order=True)
class QueueEvent:
    """
    優先度付きキューのエントリ。
    priority が小さいほど先に処理される（0が最高優先度）。
    """

    priority: int
    text: str = field(compare=False)
    interrupted_action: str | None = field(
        default=None, compare=False
    )  # 割り込み時のツール名


# 優先度定数
PRIORITY_VOICE = 0  # 音声入力（最高優先度）
PRIORITY_BEAT = 10  # ハートビート（Phase2）
