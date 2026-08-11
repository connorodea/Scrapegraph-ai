"""SQLite persistence, provenance, full-text search, and exports."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import Artifact, RepositoryRecord, ScoreBreakdown

SEARCH_TOKEN_RE = re.compile(r"[\w-]{2,}", re.UNICODE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class RadarDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RadarDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS repositories (
            github_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL UNIQUE,
            owner TEXT NOT NULL,
            name TEXT NOT NULL,
            html_url TEXT NOT NULL,
            description TEXT,
            language TEXT,
            stars INTEGER NOT NULL DEFAULT 0,
            forks INTEGER NOT NULL DEFAULT 0,
            open_issues INTEGER NOT NULL DEFAULT 0,
            size_kb INTEGER NOT NULL DEFAULT 0,
            default_branch TEXT NOT NULL DEFAULT 'main',
            license_spdx TEXT,
            topics_json TEXT NOT NULL DEFAULT '[]',
            is_fork INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            disabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            pushed_at TEXT,
            discovered_at TEXT NOT NULL,
            enriched_at TEXT,
            source_query TEXT,
            readme_text TEXT NOT NULL DEFAULT '',
            tree_paths_json TEXT NOT NULL DEFAULT '[]',
            tree_truncated INTEGER NOT NULL DEFAULT 0,
            raw_json TEXT NOT NULL DEFAULT '{}',
            score REAL NOT NULL DEFAULT 0,
            classifications_json TEXT NOT NULL DEFAULT '[]',
            score_breakdown_json TEXT NOT NULL DEFAULT '{}',
            snapshot_path TEXT,
            snapshot_sha256 TEXT,
            snapshot_commit TEXT
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL REFERENCES repositories(github_id) ON DELETE CASCADE,
            artifact_type TEXT NOT NULL,
            name TEXT,
            url TEXT,
            identifier TEXT,
            source TEXT NOT NULL,
            path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(repo_id, artifact_type, identifier, url, path)
        );
        CREATE TABLE IF NOT EXISTS query_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_query TEXT NOT NULL,
            shard_query TEXT NOT NULL,
            total_count INTEGER NOT NULL DEFAULT 0,
            fetched_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS derived_keywords (
            term TEXT PRIMARY KEY,
            score REAL NOT NULL,
            document_frequency INTEGER NOT NULL,
            relevant_document_frequency INTEGER NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS repository_fts USING fts5(
            repo_id UNINDEXED,
            full_name,
            description,
            readme_text,
            artifact_text,
            tokenize='porter unicode61'
        );
        CREATE INDEX IF NOT EXISTS idx_repositories_score ON repositories(score DESC);
        CREATE INDEX IF NOT EXISTS idx_repositories_enriched ON repositories(enriched_at);
        CREATE INDEX IF NOT EXISTS idx_artifacts_repo ON artifacts(repo_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
        """
        with self.transaction() as conn:
            conn.executescript(schema)

    def upsert_repository(self, repository: RepositoryRecord) -> bool:
        existed = self.connection.execute("SELECT 1 FROM repositories WHERE github_id=?", (repository.github_id,)).fetchone() is not None
        values = (
            repository.github_id, repository.full_name, repository.owner, repository.name,
            repository.html_url, repository.description, repository.language, repository.stars,
            repository.forks, repository.open_issues, repository.size_kb, repository.default_branch,
            repository.license_spdx, _json(repository.topics), int(repository.is_fork),
            int(repository.archived), int(repository.disabled), str(repository.created_at) if repository.created_at else None,
            str(repository.updated_at) if repository.updated_at else None, str(repository.pushed_at) if repository.pushed_at else None,
            repository.discovered_at.isoformat(), str(repository.enriched_at) if repository.enriched_at else None,
            repository.source_query, repository.readme_text, _json(repository.tree_paths), _json(repository.raw),
        )
        sql = """
        INSERT INTO repositories (
            github_id, full_name, owner, name, html_url, description, language, stars,
            forks, open_issues, size_kb, default_branch, license_spdx, topics_json,
            is_fork, archived, disabled, created_at, updated_at, pushed_at, discovered_at,
            enriched_at, source_query, readme_text, tree_paths_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(github_id) DO UPDATE SET
            full_name=excluded.full_name,
            owner=excluded.owner,
            name=excluded.name,
            html_url=excluded.html_url,
            description=excluded.description,
            language=excluded.language,
            stars=excluded.stars,
            forks=excluded.forks,
            open_issues=excluded.open_issues,
            size_kb=excluded.size_kb,
            default_branch=excluded.default_branch,
            license_spdx=COALESCE(excluded.license_spdx, repositories.license_spdx),
            topics_json=CASE WHEN excluded.topics_json != '[]' THEN excluded.topics_json ELSE repositories.topics_json END,
            is_fork=excluded.is_fork,
            archived=excluded.archived,
            disabled=excluded.disabled,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            pushed_at=excluded.pushed_at,
            source_query=COALESCE(excluded.source_query, repositories.source_query),
            readme_text=CASE WHEN excluded.readme_text != '' THEN excluded.readme_text ELSE repositories.readme_text END,
            tree_paths_json=CASE WHEN excluded.tree_paths_json != '[]' THEN excluded.tree_paths_json ELSE repositories.tree_paths_json END,
            raw_json=excluded.raw_json
        """
        with self.transaction() as conn:
            conn.execute(sql, values)
        self.refresh_fts(repository.github_id)
        return not existed

    def update_enrichment(self, repository: RepositoryRecord, *, tree_truncated: bool, artifacts: Iterable[Artifact], score: ScoreBreakdown) -> None:
        repository.enriched_at = datetime.now(timezone.utc)
        artifacts = list(artifacts)
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE repositories SET description=?, language=?, stars=?, forks=?, open_issues=?,
                    size_kb=?, default_branch=?, license_spdx=?, topics_json=?, is_fork=?, archived=?,
                    disabled=?, updated_at=?, pushed_at=?, enriched_at=?, readme_text=?,
                    tree_paths_json=?, tree_truncated=?, raw_json=?, score=?, classifications_json=?,
                    score_breakdown_json=? WHERE github_id=?
                """,
                (
                    repository.description, repository.language, repository.stars, repository.forks,
                    repository.open_issues, repository.size_kb, repository.default_branch,
                    repository.license_spdx, _json(repository.topics), int(repository.is_fork),
                    int(repository.archived), int(repository.disabled), str(repository.updated_at) if repository.updated_at else None,
                    str(repository.pushed_at) if repository.pushed_at else None, repository.enriched_at.isoformat(),
                    repository.readme_text, _json(repository.tree_paths), int(tree_truncated),
                    _json(repository.raw), score.total, _json(score.classifications), score.model_dump_json(),
                    repository.github_id,
                ),
            )
            conn.execute("DELETE FROM artifacts WHERE repo_id=?", (repository.github_id,))
            for artifact in artifacts:
                conn.execute(
                    """INSERT OR IGNORE INTO artifacts
                    (repo_id, artifact_type, name, url, identifier, source, path, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (repository.github_id, artifact.artifact_type.value, artifact.name, artifact.url,
                     artifact.identifier, artifact.source, artifact.path, _json(artifact.metadata)),
                )
        self.refresh_fts(repository.github_id)

    def refresh_fts(self, repo_id: int) -> None:
        row = self.connection.execute("SELECT full_name, description, readme_text FROM repositories WHERE github_id=?", (repo_id,)).fetchone()
        if row is None:
            return
        artifacts = self.connection.execute("SELECT artifact_type, name, url, identifier, path FROM artifacts WHERE repo_id=?", (repo_id,)).fetchall()
        artifact_text = " ".join(" ".join(str(value or "") for value in artifact) for artifact in artifacts)
        with self.transaction() as conn:
            conn.execute("DELETE FROM repository_fts WHERE repo_id=?", (str(repo_id),))
            conn.execute("INSERT INTO repository_fts (repo_id, full_name, description, readme_text, artifact_text) VALUES (?, ?, ?, ?, ?)", (str(repo_id), row["full_name"], row["description"] or "", row["readme_text"] or "", artifact_text))

    def start_query_run(self, base_query: str, shard_query: str, total_count: int) -> int:
        with self.transaction() as conn:
            cursor = conn.execute("INSERT INTO query_runs (base_query, shard_query, total_count, status, started_at) VALUES (?, ?, ?, 'running', ?)", (base_query, shard_query, total_count, _now_iso()))
        return int(cursor.lastrowid)

    def finish_query_run(self, run_id: int, *, fetched_count: int, status: str = "completed", error: str | None = None) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE query_runs SET fetched_count=?, status=?, error=?, finished_at=? WHERE id=?", (fetched_count, status, error, _now_iso(), run_id))

    def candidate_repositories(self, *, limit: int = 100, only_unenriched: bool = True, minimum_score: float = 0) -> list[dict[str, Any]]:
        clauses = ["score >= ?"]
        params: list[Any] = [minimum_score]
        if only_unenriched:
            clauses.append("enriched_at IS NULL")
        params.append(limit)
        rows = self.connection.execute(f"SELECT * FROM repositories WHERE {' AND '.join(clauses)} ORDER BY stars DESC, updated_at DESC LIMIT ?", params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def documents(self, *, minimum_score: float = 0) -> list[str]:
        rows = self.connection.execute("SELECT full_name, description, readme_text FROM repositories WHERE score >= ? AND readme_text != ''", (minimum_score,)).fetchall()
        return [" ".join([row["full_name"], row["description"] or "", row["readme_text"] or ""]) for row in rows]

    def save_derived_keywords(self, keywords: Iterable[Any]) -> None:
        with self.transaction() as conn:
            for keyword in keywords:
                conn.execute(
                    """INSERT INTO derived_keywords (term, score, document_frequency, relevant_document_frequency, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(term) DO UPDATE SET score=excluded.score,
                    document_frequency=excluded.document_frequency, relevant_document_frequency=excluded.relevant_document_frequency,
                    source=excluded.source, updated_at=excluded.updated_at""",
                    (keyword.term, keyword.score, keyword.document_frequency, keyword.relevant_document_frequency, keyword.source, _now_iso()),
                )

    def top_derived_keywords(self, *, limit: int = 50) -> list[str]:
        return [str(row["term"]) for row in self.connection.execute("SELECT term FROM derived_keywords ORDER BY score DESC LIMIT ?", (limit,)).fetchall()]

    def search(self, query: str, *, limit: int = 25, minimum_score: float = 0) -> list[dict[str, Any]]:
        tokens = SEARCH_TOKEN_RE.findall(query.casefold())
        if not tokens:
            rows = self.connection.execute("SELECT * FROM repositories WHERE score >= ? ORDER BY score DESC LIMIT ?", (minimum_score, limit)).fetchall()
            return [self._row_to_dict(row) for row in rows]
        fts_query = " AND ".join(f'"{token}"*' for token in tokens[:12])
        try:
            rows = self.connection.execute(
                """SELECT r.*, bm25(repository_fts) AS fts_rank FROM repository_fts
                JOIN repositories r ON r.github_id=CAST(repository_fts.repo_id AS INTEGER)
                WHERE repository_fts MATCH ? AND r.score >= ? ORDER BY r.score DESC, fts_rank ASC LIMIT ?""",
                (fts_query, minimum_score, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            pattern = f"%{query}%"
            rows = self.connection.execute("SELECT * FROM repositories WHERE score >= ? AND (full_name LIKE ? OR description LIKE ? OR readme_text LIKE ?) ORDER BY score DESC LIMIT ?", (minimum_score, pattern, pattern, pattern, limit)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_snapshot(self, repo_id: int, *, path: str, sha256: str, commit_sha: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE repositories SET snapshot_path=?, snapshot_sha256=?, snapshot_commit=? WHERE github_id=?", (path, sha256, commit_sha, repo_id))

    def stats(self) -> dict[str, Any]:
        return {
            "repositories": self.connection.execute("SELECT COUNT(*) FROM repositories").fetchone()[0],
            "enriched": self.connection.execute("SELECT COUNT(*) FROM repositories WHERE enriched_at IS NOT NULL").fetchone()[0],
            "artifacts": self.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            "high_priority": self.connection.execute("SELECT COUNT(*) FROM repositories WHERE score >= 55").fetchone()[0],
            "database": str(self.path),
        }

    def export(self, destination: str | Path, *, minimum_score: float = 0) -> Path:
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._row_to_dict(row) for row in self.connection.execute("SELECT * FROM repositories WHERE score >= ? ORDER BY score DESC", (minimum_score,)).fetchall()]
        if target.suffix.casefold() == ".csv":
            fieldnames = sorted({key for row in rows for key in row})
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _json(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
        else:
            with target.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(_json(row) + "\n")
        return target

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        for key in ("topics_json", "tree_paths_json", "raw_json", "classifications_json", "score_breakdown_json"):
            if key in payload:
                try:
                    payload[key.removesuffix("_json")] = json.loads(payload.pop(key) or "null")
                except json.JSONDecodeError:
                    payload[key.removesuffix("_json")] = None
        for key in ("is_fork", "archived", "disabled", "tree_truncated"):
            if key in payload:
                payload[key] = bool(payload[key])
        return payload
