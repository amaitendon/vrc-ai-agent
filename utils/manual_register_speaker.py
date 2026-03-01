"""
tests/manual_register_speaker.py

話者をマイクで録音し、SQLite に登録するCLIスクリプト。

使い方:
    # 登録
    uv run python utils/manual_register_speaker.py --label "User" --duration 5

    # 一覧
    uv run python utils/manual_register_speaker.py --list

    # 削除
    uv run python utils/manual_register_speaker.py --delete <speaker_id>
"""

import argparse
import os
import sqlite3
import sys
import time

# プロジェクトルートにパスを通す
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402
from aiavatar.device.audio import AudioRecorder, AudioDevice  # noqa: E402
from aiavatar.sts.stt.speaker_registry.base import SpeakerRegistry  # noqa: E402
from inputs.speaker_store_sqlite import SQLiteSpeakerStore  # noqa: E402
from utils.audio_device import get_device_index_by_name  # noqa: E402

load_dotenv()


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _build_registry(db_path: str, threshold: float = 0.72) -> SpeakerRegistry:
    store = SQLiteSpeakerStore(db_path)
    return SpeakerRegistry(match_threshold=threshold, store=store)


def _resolve_input_device() -> int:
    device_index_str = os.getenv("AUDIO_INPUT_DEVICE_INDEX", "")
    device_name_str = os.getenv("AUDIO_INPUT_DEVICE_NAME", "")

    if device_index_str.isdigit() and int(device_index_str) >= 0:
        return int(device_index_str)
    elif device_name_str:
        return get_device_index_by_name(device_name_str, is_input=True)
    return -1


async def _record_pcm(duration: float, sample_rate: int, device_index: int) -> bytes:
    """マイクから PCM (int16 mono) を録音して bytes で返す（async 対応）。"""
    resolved = AudioDevice(input_device=device_index).input_device
    recorder = AudioRecorder(sample_rate=sample_rate, device_index=resolved)

    print(f"  録音開始します（{duration:.1f} 秒）。話しかけてください...")
    await asyncio.sleep(0.3)  # 少し待ってから

    chunks: list[bytes] = []

    start = time.monotonic()
    async for chunk in recorder.start_stream():
        chunks.append(chunk)
        elapsed = time.monotonic() - start
        remaining = duration - elapsed
        bars = int(elapsed / duration * 20)
        print(
            f"\r  [{'#' * bars}{'.' * (20 - bars)}] {remaining:.1f}s 残り",
            end="",
            flush=True,
        )
        if elapsed >= duration:
            recorder.stop_stream()
            break

    print()  # 改行
    return b"".join(chunks)


# ── サブコマンド ───────────────────────────────────────────────────────────────


async def cmd_register(args):
    db_path = os.getenv("SPEAKER_GATE_DB_PATH", "data/speakers.db")
    label = args.label
    duration = float(args.duration)

    registry = _build_registry(db_path)
    device_index = _resolve_input_device()

    # resemblyzer の sample_rate は 16000 固定
    sample_rate = 16000

    print(f"\n[登録] ラベル: {label!r}  録音時間: {duration}秒  DB: {db_path}")
    audio_bytes = await _record_pcm(duration, sample_rate, device_index)

    print("  埋め込みを生成中...")
    result = await asyncio.to_thread(
        registry.match_topk_from_pcm, audio_bytes, sample_rate
    )

    speaker_id = result.chosen.speaker_id
    # ラベルをメタデータに保存
    registry.set_metadata(speaker_id, "label", label)

    if result.chosen.is_new:
        print(f"  新規登録完了: {speaker_id}  (label={label!r})")
    else:
        # 既存にマッチした場合はラベルを上書きして登録更新
        print(
            f"  既存の話者 {speaker_id} に類似（sim={result.chosen.similarity:.3f}）。ラベルを更新しました。"
        )


def cmd_list(args):
    db_path = os.getenv("SPEAKER_GATE_DB_PATH", "data/speakers.db")
    store = SQLiteSpeakerStore(db_path)

    total = store.count()
    print(f"\n[登録済み話者一覧]  DB: {db_path}  合計: {total} 件\n")

    if total == 0:
        print("  (登録なし)")
        return

    print(f"  {'ID':<30}  ラベル")
    print(f"  {'-' * 30}  ------")
    for sid, _emb, meta in store.all_items():
        label = meta.get("label", "(未設定)")
        print(f"  {sid:<30}  {label}")


def cmd_delete(args):
    db_path = os.getenv("SPEAKER_GATE_DB_PATH", "data/speakers.db")
    speaker_id = args.delete

    store = SQLiteSpeakerStore(db_path)
    try:
        store.get(speaker_id)  # 存在確認
    except KeyError:
        print(f"[エラー] ID が見つかりません: {speaker_id}")
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
    print(f"[削除完了] {speaker_id}")


# ── エントリーポイント ────────────────────────────────────────────────────────


async def _main():
    parser = argparse.ArgumentParser(
        description="話者識別 — 登録・一覧・削除スクリプト"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--label", metavar="LABEL", help="マイクで録音して話者を登録する"
    )
    group.add_argument(
        "--list", action="store_true", help="登録済み話者の一覧を表示する"
    )
    group.add_argument("--delete", metavar="SPEAKER_ID", help="指定した話者を削除する")

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="録音時間（秒）。--label 指定時のみ有効。デフォルト: 5",
    )

    args = parser.parse_args()

    if args.label:
        await cmd_register(args)
    elif args.list:
        cmd_list(args)
    elif args.delete:
        cmd_delete(args)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
