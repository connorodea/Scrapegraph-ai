"""SQLite persistence, run history, and full-text search."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import ResearchCandidate, RunEvent, RunKind, RunSummary


class RadarStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fts_enabled = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_kind TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    saved_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, candidate_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_run_score
                    ON candidates(run_id, score DESC);
                CREATE INDEX IF NOT EXISTS idx_candidates_kind
                    ON candidates(kind);

                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    level TEXT NOT NULL,
                    progress REAL NOT NULL,
                    plain_english TEXT NOT NULL,
                    technical_detail TEXT NOT NULL,
                    counters_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                    ON events(run_id, sequence);
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS candidate_fts USING fts5(
                        run_id UNINDEXED,
                        candidate_id UNINDEXED,
                        title,
                        body,
                        tokenize='porter unicode61'
                    )
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

    def create_run(self, summary: RunSummary) -> None:
        payload = json.dumps(summary.model_dump(mode="json"), ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, run_kind, started_at, completed_at, status, summary_json)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_kind=excluded.run_kind,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    status=excluded.status,
                    summary_json=excluded.summary_json
                """,
                (
                    summary.run_id,
                    summary.run_kind.value,
                    summary.started_at.isoformat(),
                    summary.completed_at.isoformat() if summary.completed_at else None,
                    summary.status,
                    payload,
                ),
            )

    def update_run(self, summary: RunSummary) -> None:
        self.create_run(summary)

    def save_event(self, event: RunEvent) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(
                    run_id, timestamp, stage, level, progress,
                    plain_english, technical_detail, counters_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.timestamp.isoformat(),
                    event.stage.value,
                    event.level.value,
                    event.progress,
                    event.plain_english,
                    event.technical_detail,
                    json.dumps(event.counters, ensure_ascii=False),
                ),
            )

    def save_candidates(
        self, run_id: str, candidates: Iterable[ResearchCandidate]
    ) -> int:
        rows = list(candidates)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            for candidate in rows:
                payload = json.dumps(
                    candidate.model_dump(mode="json"), ensure_ascii=False
                )
                conn.execute(
                    """
                    INSERT INTO candidates(
                        run_id, candidate_id, kind, source, title, url,
                        score, payload_json, saved_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                        kind=excluded.kind,
                        source=excluded.source,
                        title=excluded.title,
                        url=excluded.url,
                        score=excluded.score,
                        payload_json=excluded.payload_json,
                        saved_at=excluded.saved_at
                    """,
                    (
                        run_id,
                        candidate.candidate_id,
                        candidate.kind.value,
                        candidate.source,
                        candidate.title,
                        candidate.url,
                        candidate.score.overall,
                        payload,
                        now,
                    ),
                )
                if self._fts_enabled:
                    conn.execute(
                        "DELETE FROM candidate_fts WHERE run_id=? AND candidate_id=?",
                        (run_id, candidate.candidate_id),
                    )
                    conn.execute(
                        "INSERT INTO candidate_fts(run_id, candidate_id, title, body) VALUES(?, ?, ?, ?)",
                        (
                            run_id,
                            candidate.candidate_id,
                            candidate.title,
                            candidate.searchable_text,
                        ),
                    )
        return len(rows)

    def list_runs(self, *, limit: int = 50) -> list[RunSummary]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT summary_json FROM runs ORDER BY started_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [RunSummary.model_validate_json(row["summary_json"]) for row in rows]

    def get_run(self, run_id: str) -> RunSummary | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return RunSummary.model_validate_json(row["summary_json"]) if row else None

    def list_candidates(
        self,
        run_id: str | None = None,
        *,
        limit: int = 1000,
        minimum_score: float = 0.0,
    ) -> list[ResearchCandidate]:
        query = "SELECT payload_json FROM candidates WHERE score >= ?"
        params: list[object] = [minimum_score]
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY score DESC LIMIT ?"
        params.append(max(1, limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [ResearchCandidate.model_validate_json(row["payload_json"]) for row in rows]

    def list_events(self, run_id: str, *, limit: int = 5000) -> list[RunEvent]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, stage, level, progress, plain_english,
                       technical_detail, counters_json
                FROM events WHERE run_id=? ORDER BY sequence ASC LIMIT ?
                """,
                (run_id, max(1, limit)),
            ).fetchall()
        run = self.get_run(run_id)
        run_kind = run.run_kind if run else RunKind.GITHUB
        return [
            RunEvent(
                run_id=run_id,
                run_kind=run_kind,
                timestamp=row["timestamp"],
                stage=row["stage"],
                level=row["level"],
                progress=float(row["progress"]),
                plain_english=row["plain_english"],
                technical_detail=row["technical_detail"],
                counters=json.loads(row["counters_json"]),
            )
            for row in rows
        ]

    def search(
        self,
        text: str,
        *,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[ResearchCandidate]:
        term = text.strip()
        if not term:
            return self.list_candidates(run_id, limit=limit)
        with self._lock, self._connect() as conn:
            if self._fts_enabled:
                sql = (
                    "SELECT c.payload_json FROM candidate_fts f "
                    "JOIN candidates c ON c.run_id=f.run_id AND c.candidate_id=f.candidate_id "
                    "WHERE candidate_fts MATCH ?"
                )
                params: list[object] = [term]
                if run_id:
                    sql += " AND c.run_id=?"
                    params.append(run_id)
                sql += " ORDER BY bm25(candidate_fts), c.score DESC LIMIT ?"
                params.append(max(1, limit))
                try:
                    rows = conn.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            else:
                rows = []
            if not rows:
                like = f"%{term}%"
                sql = "SELECT payload_json FROM candidates WHERE payload_json LIKE ?"
                params = [like]
                if run_id:
                    sql += " AND run_id=?"
                    params.append(run_id)
                sql += " ORDER BY score DESC LIMIT ?"
                params.append(max(1, limit))
                rows = conn.execute(sql, params).fetchall()
        return [ResearchCandidate.model_validate_json(row["payload_json"]) for row in rows]

    def delete_run(self, run_id: str) -> None:
        with self._lock, self._connect() as conn:
            if self._fts_enabled:
                conn.execute("DELETE FROM candidate_fts WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
