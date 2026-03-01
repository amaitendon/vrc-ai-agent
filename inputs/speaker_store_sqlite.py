"""
inputs/speaker_store_sqlite.py

SQLite バックエンドの BaseSpeakerStore 実装。
音声埋め込み（numpy float32 ベクトル）を BLOB として保存し、
Top-K マッチングは Python 側でコサイン類似度を計算する（小規模利用前提）。
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    def __init__(self, db_path: str, embedding_dim: Optional[int] = None) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        # DB ファイルの親ディレクトリを自動生成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        try:
            self._init_db()
        except Exception:
            self._conn.close()
            raise

    # ── DB 初期化 ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS speakers (
                        id        TEXT PRIMARY KEY,
                        embedding BLOB    NOT NULL,
                        dim       INTEGER NOT NULL,
                        metadata  TEXT    NOT NULL DEFAULT '{}'
                    )
                """)
                if self.embedding_dim is not None:
                    row = self._conn.execute(
                        "SELECT dim FROM speakers LIMIT 1"
                    ).fetchone()
                    if row is not None and row["dim"] != self.embedding_dim:
                        raise ValueError(
                            f"DB has existing embeddings with dim={row['dim']}, "
                            f"but embedding_dim={self.embedding_dim} was specified."
                        )

    def close(self) -> None:
        """接続を閉じる"""
        with self._lock:
            self._conn.close()

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
        if self.embedding_dim is not None and embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embedding.shape[0]}"
            )

        emb_f32 = embedding.astype(np.float32, copy=False)
        blob = emb_f32.tobytes()
        dim = emb_f32.shape[0]
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            with self._conn:
                # 既存行がある場合: embedding は新値で上書き、metadata は既存値にマージ
                existing = self._conn.execute(
                    "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
                ).fetchone()

                if existing is not None:
                    old_meta = json.loads(existing["metadata"] or "{}")
                    if metadata:
                        old_meta.update(metadata)
                    meta_json = json.dumps(old_meta, ensure_ascii=False)
                    self._conn.execute(
                        "UPDATE speakers SET embedding=?, dim=?, metadata=? WHERE id=?",
                        (blob, dim, meta_json, speaker_id),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO speakers (id, embedding, dim, metadata) VALUES (?, ?, ?, ?)",
                        (speaker_id, blob, dim, meta_json),
                    )

    def get(self, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding, dim, metadata FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        emb = self._from_blob(row["embedding"], row["dim"])
        md = json.loads(row["metadata"] or "{}")
        return emb, md

    def set_metadata(self, speaker_id: str, key: str, value: Any) -> None:
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown speaker_id: {speaker_id}")
                md = json.loads(row["metadata"] or "{}")
                md[key] = value
                self._conn.execute(
                    "UPDATE speakers SET metadata=? WHERE id=?",
                    (json.dumps(md, ensure_ascii=False), speaker_id),
                )

    def get_metadata(self, speaker_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown speaker_id: {speaker_id}")
        md = json.loads(row["metadata"] or "{}")
        return md.get(key, default)

    def all_items(self) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding, dim, metadata FROM speakers"
            ).fetchall()

        results = []
        for row in rows:
            emb = self._from_blob(row["embedding"], row["dim"])
            md = json.loads(row["metadata"] or "{}")
            results.append((row["id"], emb, md))
        return results

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM speakers").fetchone()
        return int(row["cnt"])

    def topk_similarity(self, q_norm: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        全件取り出してコサイン類似度を Python 側で計算し、Top-K を返す。
        話者数が少ない前提なので全件スキャンで問題ない。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding, dim FROM speakers"
            ).fetchall()

        if not rows:
            raise RuntimeError("SQLiteSpeakerStore is empty.")

        sids = []
        embs = []
        for row in rows:
            sids.append(row["id"])
            embs.append(self._from_blob(row["embedding"], row["dim"]))

        matrix = np.vstack(embs).astype(np.float32)  # (N, D)

        # コサイン類似度のために正規化
        q = q_norm.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        matrix_normed = matrix / norms

        sims = matrix_normed @ q  # (N,)

        k = max(1, min(k, len(sids)))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(sids[i], float(sims[i])) for i in idx]
