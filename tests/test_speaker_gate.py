"""
tests/test_speaker_gate.py

SQLiteSpeakerStore と SpeakerRegistry のユニットテスト。
VoiceEncoder を unittest.mock でモックし、重い resemblyzer 推論を回避する。
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inputs.speaker_store_sqlite import SQLiteSpeakerStore


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _rand_emb(dim: int = 256) -> np.ndarray:
    """ランダムな L2 正規化済み埋め込みを生成する。"""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


@pytest.fixture
def tmp_db(tmp_path) -> SQLiteSpeakerStore:
    """テストごとに一時 SQLite ファイルを使用する。"""
    db_path = str(tmp_path / "test_speakers.db")
    return SQLiteSpeakerStore(db_path)


# ── SQLiteSpeakerStore テスト ────────────────────────────────────────────────


def test_sqlite_store_initial_count_is_zero(tmp_db):
    """新規DB は count=0 であること。"""
    assert tmp_db.count() == 0


def test_sqlite_store_upsert_and_count(tmp_db):
    """upsert すると count が増えること。"""
    emb = _rand_emb()
    tmp_db.upsert("spk_aaa", emb, metadata={"label": "alice"})
    assert tmp_db.count() == 1

    tmp_db.upsert("spk_bbb", _rand_emb(), metadata={"label": "bob"})
    assert tmp_db.count() == 2


def test_sqlite_store_upsert_is_idempotent(tmp_db):
    """同じ ID で upsert を 2 回しても count が増えないこと。"""
    emb = _rand_emb()
    tmp_db.upsert("spk_aaa", emb)
    tmp_db.upsert("spk_aaa", emb, metadata={"label": "alice"})
    assert tmp_db.count() == 1


def test_sqlite_store_get(tmp_db):
    """get で保存した埋め込みとメタデータを取得できること。"""
    emb = _rand_emb()
    tmp_db.upsert("spk_aaa", emb, metadata={"label": "alice"})
    retrieved_emb, meta = tmp_db.get("spk_aaa")

    np.testing.assert_allclose(retrieved_emb, emb, atol=1e-6)
    assert meta["label"] == "alice"


def test_sqlite_store_get_unknown_raises(tmp_db):
    """存在しない ID で get すると KeyError が発生すること。"""
    with pytest.raises(KeyError):
        tmp_db.get("no_such_id")


def test_sqlite_store_set_and_get_metadata(tmp_db):
    """set_metadata / get_metadata が正しく動くこと。"""
    tmp_db.upsert("spk_aaa", _rand_emb())
    tmp_db.set_metadata("spk_aaa", "label", "alice")
    assert tmp_db.get_metadata("spk_aaa", "label") == "alice"
    assert tmp_db.get_metadata("spk_aaa", "missing_key", "default") == "default"


def test_sqlite_store_topk_returns_correct_order(tmp_db):
    """
    クエリに近い埋め込みが Top-1 に来ること。
    spk_a と全く同じベクトルを query として渡すと spk_a が Top-1 になるはず。
    """
    emb_a = _rand_emb()
    emb_b = _rand_emb()
    emb_c = _rand_emb()
    tmp_db.upsert("spk_a", emb_a)
    tmp_db.upsert("spk_b", emb_b)
    tmp_db.upsert("spk_c", emb_c)

    results = tmp_db.topk_similarity(emb_a, k=3)
    # 先頭が spk_a で similarity ≈ 1.0
    assert results[0][0] == "spk_a"
    assert abs(results[0][1] - 1.0) < 1e-5


def test_sqlite_store_topk_raises_when_empty(tmp_db):
    """空の DB で topk_similarity を呼ぶと RuntimeError になること。"""
    with pytest.raises(RuntimeError):
        tmp_db.topk_similarity(_rand_emb(), k=1)


def test_sqlite_store_all_items(tmp_db):
    """all_items がすべての行を返すこと。"""
    tmp_db.upsert("spk_aaa", _rand_emb(), metadata={"label": "alice"})
    tmp_db.upsert("spk_bbb", _rand_emb(), metadata={"label": "bob"})
    items = list(tmp_db.all_items())
    ids = {item[0] for item in items}
    assert ids == {"spk_aaa", "spk_bbb"}


# ── SpeakerRegistry テスト（VoiceEncoder をモック） ─────────────────────────


def _make_registry(tmp_db: SQLiteSpeakerStore, threshold: float = 0.72):
    """VoiceEncoder をモックした SpeakerRegistry を返す。"""
    from aiavatar.sts.stt.speaker_registry.base import SpeakerRegistry

    registry = SpeakerRegistry(match_threshold=threshold, store=tmp_db)
    # resemblyzer の重いモデルをモックに差し替え
    registry._enc = MagicMock()
    return registry


def test_registry_first_speaker_is_registered_as_new(tmp_db):
    """初めて呼んだとき is_new=True で登録されること。"""
    registry = _make_registry(tmp_db)
    emb = _rand_emb()
    registry._enc.embed_utterance = MagicMock(return_value=emb)

    # _embed_pcm をモック（VAD から来る PCM bytes は適当でよい）
    with patch.object(registry, "_embed_pcm", return_value=emb):
        result = registry.match_topk_from_pcm(b"\x00" * 100, sample_rate=16000)

    assert result.chosen.is_new is True
    assert tmp_db.count() == 1


def test_registry_match_known_speaker(tmp_db):
    """登録済み話者は is_new=False で、高い類似度で返ること。"""
    registry = _make_registry(tmp_db)
    emb_alice = _rand_emb()

    # 先に登録
    with patch.object(registry, "_embed_pcm", return_value=emb_alice):
        first = registry.match_topk_from_pcm(b"\x00" * 100, sample_rate=16000)
    alice_id = first.chosen.speaker_id
    tmp_db.set_metadata(alice_id, "label", "alice")

    # 同じ埋め込みで再度マッチング
    with patch.object(registry, "_embed_pcm", return_value=emb_alice):
        result = registry.match_topk_from_pcm(b"\x00" * 100, sample_rate=16000)

    assert result.chosen.is_new is False
    assert result.chosen.speaker_id == alice_id
    assert result.chosen.similarity > 0.99


def test_registry_reject_unknown_speaker(tmp_db):
    """
    既存話者と大きく異なる埋め込みは is_new=True（新規登録）になること。
    threshold=0.99 にして、わずかに違うベクトルも弾かれるようにする。
    """
    registry = _make_registry(tmp_db, threshold=0.99)

    emb_alice = _rand_emb()
    emb_unknown = _rand_emb()  # まったく別のランダムベクトル

    # alice を登録
    with patch.object(registry, "_embed_pcm", return_value=emb_alice):
        registry.match_topk_from_pcm(b"\x00" * 100, sample_rate=16000)

    # 別人で照合 → 新規扱い
    with patch.object(registry, "_embed_pcm", return_value=emb_unknown):
        result = registry.match_topk_from_pcm(b"\x00" * 100, sample_rate=16000)

    assert result.chosen.is_new is True
