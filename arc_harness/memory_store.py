"""Structured, searchable memory store for ARC harness."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .events import utc_now


MEMORY_CATEGORIES = {
    "fact",
    "failure",
    "correction",
    "insight",
    "preference",
    "convention",
    "tool-quirk",
    "procedure",
    "episode",
    "rule",
}


@dataclass(frozen=True)
class SearchResult:
    memory_id: str
    text: str
    category: str
    namespace: tuple[str, ...]
    scope: str
    tags: tuple[str, ...]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "category": self.category,
            "namespace": list(self.namespace),
            "scope": self.scope,
            "tags": list(self.tags),
            "score": self.score,
            "metadata": self.metadata,
        }


class LightweightEmbeddingIndex:
    """Hashing-vector index used when no external embedding model is available."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def similarity(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))


class StructuredMemoryStore:
    """SQLite memory store with namespace, FTS, and lightweight vector search."""

    def __init__(self, path: str | Path, embedding_index: LightweightEmbeddingIndex | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_index = embedding_index or LightweightEmbeddingIndex()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                scope TEXT NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                source_episode_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_vectors (
                memory_id TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id)
            )
            """
        )
        self.fts_enabled = self._setup_fts()
        self.conn.commit()

    def _setup_fts(self) -> bool:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(memory_id UNINDEXED, text, category, tags, content='')
                """
            )
            return True
        except sqlite3.OperationalError:
            return False

    def put(
        self,
        *,
        text: str,
        category: str = "fact",
        namespace: Iterable[str] = ("global",),
        key: str | None = None,
        scope: str = "global",
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        source_episode_id: str | None = None,
    ) -> str:
        if category not in MEMORY_CATEGORIES:
            category = "fact"
        memory_id = key or uuid.uuid4().hex
        namespace_tuple = tuple(namespace)
        now = utc_now()
        tag_tuple = tuple(tags)
        metadata = dict(metadata or {})
        vector = self.embedding_index.embed(text)
        self.conn.execute(
            """
            INSERT INTO memories (
                memory_id, namespace, key, text, category, scope, tags, metadata,
                confidence, importance, source_episode_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                text=excluded.text,
                category=excluded.category,
                scope=excluded.scope,
                tags=excluded.tags,
                metadata=excluded.metadata,
                confidence=excluded.confidence,
                importance=excluded.importance,
                source_episode_id=excluded.source_episode_id,
                updated_at=excluded.updated_at
            """,
            (
                memory_id,
                _encode_namespace(namespace_tuple),
                memory_id,
                text,
                category,
                scope,
                json.dumps(tag_tuple, ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False),
                confidence,
                importance,
                source_episode_id,
                now,
                now,
            ),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_vectors(memory_id, vector) VALUES (?, ?)",
            (memory_id, json.dumps(vector)),
        )
        if self.fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            self.conn.execute(
                "INSERT INTO memories_fts(memory_id, text, category, tags) VALUES (?, ?, ?, ?)",
                (memory_id, text, category, " ".join(tag_tuple)),
            )
        self.conn.commit()
        return memory_id

    def search(
        self,
        query: str,
        *,
        namespace: Iterable[str] | None = None,
        category: str | None = None,
        tags: Iterable[str] = (),
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        rows = self._candidate_rows(query, namespace=namespace, category=category, tags=tags, mode=mode)
        query_vector = self.embedding_index.embed(query)
        terms = set(_tokens(query))
        tag_filter = set(tags)
        results = []
        for row in rows:
            text = row["text"]
            row_tags = tuple(json.loads(row["tags"]))
            if tag_filter and not tag_filter.intersection(row_tags):
                continue
            keyword_score = _keyword_score(terms, text)
            vector_row = self.conn.execute("SELECT vector FROM memory_vectors WHERE memory_id = ?", (row["memory_id"],)).fetchone()
            vector_score = 0.0
            if vector_row:
                vector_score = self.embedding_index.similarity(query_vector, json.loads(vector_row["vector"]))
            score = _combine_score(keyword_score, vector_score, float(row["importance"]), float(row["confidence"]), mode)
            results.append(
                SearchResult(
                    memory_id=row["memory_id"],
                    text=text,
                    category=row["category"],
                    namespace=_decode_namespace(row["namespace"]),
                    scope=row["scope"],
                    tags=row_tags,
                    score=score,
                    metadata=json.loads(row["metadata"]),
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        selected = results[:limit]
        for result in selected:
            self._mark_used(result.memory_id)
        self.conn.commit()
        return selected

    def _candidate_rows(
        self,
        query: str,
        *,
        namespace: Iterable[str] | None,
        category: str | None,
        tags: Iterable[str],
        mode: str,
    ) -> list[sqlite3.Row]:
        filters = []
        params: list[Any] = []
        if namespace is not None:
            ns_prefix = _encode_namespace(tuple(namespace))
            filters.append("namespace LIKE ?")
            params.append(f"{ns_prefix}%")
        if category is not None:
            filters.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        if self.fts_enabled and query.strip() and mode in {"keyword", "hybrid"}:
            try:
                fts_rows = self.conn.execute(
                    f"""
                    SELECT m.* FROM memories_fts f
                    JOIN memories m ON m.memory_id = f.memory_id
                    {where + (' AND' if where else 'WHERE')} memories_fts MATCH ?
                    """,
                    [*params, _fts_query(query)],
                ).fetchall()
                if fts_rows:
                    return list(fts_rows)
            except sqlite3.OperationalError:
                pass
        return list(self.conn.execute(f"SELECT * FROM memories {where}", params).fetchall())

    def consolidate(self, *, category: str | None = None, max_entries: int = 200) -> int:
        """Merge low-importance duplicate-ish memories into compact summaries."""
        filters = []
        params = []
        if category:
            filters.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self.conn.execute(f"SELECT * FROM memories {where} ORDER BY updated_at DESC", params).fetchall()
        if len(rows) <= max_entries:
            return 0
        overflow = rows[max_entries:]
        by_category: dict[str, list[str]] = {}
        for row in overflow:
            by_category.setdefault(row["category"], []).append(row["text"])
        merged = 0
        for cat, texts in by_category.items():
            summary = " | ".join(text[:160] for text in texts[:20])
            self.put(
                text=f"Consolidated {len(texts)} {cat} memories: {summary}",
                category=cat,
                namespace=("global", "consolidated"),
                scope="global",
                tags=("consolidated", cat),
                importance=0.4,
            )
            merged += len(texts)
        ids = [row["memory_id"] for row in overflow]
        self.conn.executemany("DELETE FROM memories WHERE memory_id = ?", [(memory_id,) for memory_id in ids])
        self.conn.executemany("DELETE FROM memory_vectors WHERE memory_id = ?", [(memory_id,) for memory_id in ids])
        if self.fts_enabled:
            self.conn.executemany("DELETE FROM memories_fts WHERE memory_id = ?", [(memory_id,) for memory_id in ids])
        self.conn.commit()
        return merged

    def _mark_used(self, memory_id: str) -> None:
        self.conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_used_at = ? WHERE memory_id = ?",
            (utc_now(), memory_id),
        )


def _tokens(text: str) -> list[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in normalized.split() if token]


def _keyword_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(_tokens(text))
    return len(query_terms.intersection(text_terms)) / len(query_terms)


def _combine_score(keyword: float, vector: float, importance: float, confidence: float, mode: str) -> float:
    if mode == "keyword":
        base = keyword
    elif mode == "vector":
        base = vector
    else:
        base = 0.55 * keyword + 0.45 * vector
    return base * (0.7 + 0.2 * importance + 0.1 * confidence)


def _encode_namespace(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace)


def _decode_namespace(namespace: str) -> tuple[str, ...]:
    return tuple(part for part in namespace.split("/") if part)


def _fts_query(query: str) -> str:
    return " OR ".join(_tokens(query)) or query

