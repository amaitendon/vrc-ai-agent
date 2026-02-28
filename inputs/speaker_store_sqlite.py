"""
inputs/speaker_store_sqlite.py

SQLite バックエンドの BaseSpeakerStore 実装。
音声埋め込み（numpy float32 ベクトル）を BLOB として保存し、
Top-K マッチングは Python 側でコサイン類似度を計算する（小規模利用前提）。
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from aiavatar.sts.stt.speaker_registry.base import BaseSpeakerStore


class SQLiteSpeakerStore(BaseSpeakerStore):
    """
    SQLite を使用した話者ストア。

    テーブル構造:
        speakers (
            id       TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,    -- float32 numpy array の tobytes()
            dim       INTEGER NOT NULL, -- 埋め込みの次元数
            metadata  TEXT NOT NULL     -- JSON 文字列
        )
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # DB ファイルの親ディレクトリを自動生成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── DB 初期化 ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    id        TEXT PRIMARY KEY,
                    embedding BLOB    NOT NULL,
                    dim       INTEGER NOT NULL,
                    metadata  TEXT    NOT NULL DEFAULT '{}'
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── ヘルパー ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_blob(arr: np.ndarray) -> bytes:
        return arr.astype(np.float32, copy=False).tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(dim)

    # ── BaseSpeakerStore 実装 ─────────────────────────────────────────────────

    def upsert(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        emb_f32 = embedding.astype(np.float32, copy=False)
        blob = emb_f32.tobytes()
        dim = emb_f32.shape[0]
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            # 既存行がある場合は embedding と metadata をマージ更新
            existing = conn.execute(
                "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()

            if existing is not None:
                old_meta = json.loads(existing["metadata"] or "{}")
                if metadata:
                    old_meta.update(metadata)
                meta_json = json.dumps(old_meta, ensure_ascii=False)
                conn.execute(
                    "UPDATE speakers SET embedding=?, dim=?, metadata=? WHERE id=?",
                    (blob, dim, meta_json, speaker_id),
                )
            else:
                conn.execute(
                    "INSERT INTO speakers (id, embedding, dim, metadata) VALUES (?, ?, ?, ?)",
                    (speaker_id, blob, dim, meta_json),
                )

    def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding, dim, metadata FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        emb = self._from_blob(row["embedding"], row["dim"])
        md = json.loads(row["metadata"] or "{}")
        return emb, md

    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown speaker_id: {speaker_id}")
            md = json.loads(row["metadata"] or "{}")
            md[key] = value
            conn.execute(
                "UPDATE speakers SET metadata=? WHERE id=?",
                (json.dumps(md, ensure_ascii=False), speaker_id),
            )

    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        md = json.loads(row["metadata"] or "{}")
        return md.get(key, default)

    def all_items(self) -> Iterable[Tuple[str, np.ndarray, Dict[str, Any]]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding, dim, metadata FROM speakers"
            ).fetchall()
        for row in rows:
            emb = self._from_blob(row["embedding"], row["dim"])
            md = json.loads(row["metadata"] or "{}")
            yield row["id"], emb, md

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM speakers").fetchone()
        return int(row["cnt"])

    def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        全件取り出してコサイン類似度を Python 側で計算し、Top-K を返す。
        話者数が少ない前提なので全件スキャンで問題ない。
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT id, embedding, dim FROM speakers").fetchall()

        if not rows:
            raise RuntimeError("SQLiteSpeakerStore is empty.")

        sids = []
        embs = []
        for row in rows:
            sids.append(row["id"])
            embs.append(self._from_blob(row["embedding"], row["dim"]))

        matrix = np.vstack(embs).astype(np.float32)  # (N, D)
        sims = matrix @ q_norm.astype(np.float32)  # (N,)

        k = max(1, min(k, len(sids)))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(sids[i], float(sims[i])) for i in idx]
