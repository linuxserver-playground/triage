"""SQLite in the data dir: what the bot has seen and answered, and what every model run cost."""

import datetime
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS threads (
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    kind TEXT NOT NULL,
    seen_updated_at TEXT NOT NULL,
    replies INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL,
    PRIMARY KEY (repo, number)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    mode TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    usd REAL NOT NULL,
    outcome TEXT NOT NULL,
    comment_url TEXT
);
"""


def utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Thread:
    repo: str
    number: int
    kind: str
    seen_updated_at: str
    replies: int
    outcome: str


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0


class State:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def started_at(self) -> str:
        """When the bot first ran. Anything opened before it is never answered: no backfill."""
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'started_at'").fetchone()
            if row:
                return str(row["value"])
            now = utcnow()
            conn.execute("INSERT INTO meta (key, value) VALUES ('started_at', ?)", (now,))
            return now

    def thread(self, repo: str, number: int) -> Thread | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM threads WHERE repo = ? AND number = ?", (repo, number)).fetchone()
        return Thread(**dict(row)) if row else None

    def mark(self, repo: str, number: int, kind: str, updated_at: str, outcome: str, replied: bool = False) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO threads (repo, number, kind, seen_updated_at, replies, outcome) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (repo, number) DO UPDATE SET seen_updated_at = excluded.seen_updated_at, "
                "replies = threads.replies + excluded.replies, outcome = excluded.outcome",
                (repo, number, kind, updated_at, int(replied), outcome),
            )

    def record_run(
        self, repo: str, number: int, mode: str, usage: Usage, outcome: str, comment_url: str | None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs (at, repo, number, mode, model, input_tokens, output_tokens, cache_read_tokens, "
                "cache_write_tokens, usd, outcome, comment_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utcnow(),
                    repo,
                    number,
                    mode,
                    usage.model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_tokens,
                    usage.cache_write_tokens,
                    usage.usd,
                    outcome,
                    comment_url,
                ),
            )

    def month_spend(self, now: datetime.datetime | None = None) -> float:
        month = (now or datetime.datetime.now(datetime.UTC)).strftime("%Y-%m")
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(usd), 0) AS usd FROM runs WHERE at LIKE ?", (f"{month}%",)
            ).fetchone()
        return float(row["usd"])
