"""Per-talk records: who ran what, for how long, and what it cost.

One row per talk, created when the presenter names it and updated as the
session runs, so a row is useful even if the process dies mid-talk.

SQLite through the stdlib, with every call pushed to a thread: the writes are
tiny, but a blocked event loop shows up as stuttering subtitles, and that is too
high a price for bookkeeping. One connection, guarded by a lock, because the
writes are serialised anyway and WAL plus a short busy timeout handles anyone
reading the file from outside.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS talks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL,
    started_at    REAL    NOT NULL,
    ended_at      REAL,
    ip            TEXT,
    user_agent    TEXT,
    engine        TEXT,
    mode          TEXT,
    direction     TEXT,
    asr_model     TEXT,
    mt_model      TEXT,
    audio_s       REAL    NOT NULL DEFAULT 0,
    asr_requests  INTEGER NOT NULL DEFAULT 0,
    in_tok        INTEGER NOT NULL DEFAULT 0,
    out_tok       INTEGER NOT NULL DEFAULT 0,
    mt_requests   INTEGER NOT NULL DEFAULT 0,
    usd           REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS talks_started_at ON talks (started_at DESC);
"""

_USAGE_FIELDS = ("audio_s", "asr_requests", "in_tok", "out_tok", "mt_requests", "usd")


class Db:
    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            # Relative to the project, not to wherever uvicorn was launched.
            self.path = Path(__file__).resolve().parent.parent / self.path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        log.info("talk records: %s", self.path)

    # -- blocking side ------------------------------------------------------

    def _start(self, **row) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO talks ({cols}) VALUES ({marks})", tuple(row.values())
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def _update_usage(self, talk_id: int, usage: dict) -> None:
        sets = ", ".join(f"{f} = ?" for f in _USAGE_FIELDS)
        values = [usage.get(f, 0) for f in _USAGE_FIELDS]
        with self._lock:
            self._conn.execute(f"UPDATE talks SET {sets} WHERE id = ?", (*values, talk_id))
            self._conn.commit()

    def _end(self, talk_id: int, usage: dict | None) -> None:
        sets = ["ended_at = ?"]
        values: list = [time.time()]
        if usage:
            sets += [f"{f} = ?" for f in _USAGE_FIELDS]
            values += [usage.get(f, 0) for f in _USAGE_FIELDS]
        with self._lock:
            self._conn.execute(
                f"UPDATE talks SET {', '.join(sets)} WHERE id = ?", (*values, talk_id)
            )
            self._conn.commit()

    def _resume(self, talk_id: int) -> dict | None:
        """Reopen a row after a reconnect. None if it isn't there any more."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM talks WHERE id = ?", (talk_id,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute("UPDATE talks SET ended_at = NULL WHERE id = ?", (talk_id,))
            self._conn.commit()
        return dict(row)

    def _rename(self, talk_id: int, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE talks SET title = ? WHERE id = ?", (title, talk_id)
            )
            self._conn.commit()

    def _recent(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM talks ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- async API ----------------------------------------------------------

    async def start_talk(
        self,
        *,
        title: str,
        ip: str | None,
        user_agent: str | None,
        engine: str | None,
        mode: str,
        direction: str,
        asr_model: str | None,
        mt_model: str | None,
    ) -> int:
        return await asyncio.to_thread(
            self._start,
            title=title.strip()[:200] or "untitled",
            started_at=time.time(),
            ip=ip,
            user_agent=(user_agent or "")[:300] or None,
            engine=engine,
            mode=mode,
            direction=direction,
            asr_model=asr_model,
            mt_model=mt_model,
        )

    async def resume_talk(self, talk_id: int) -> dict | None:
        return await asyncio.to_thread(self._resume, talk_id)

    async def rename_talk(self, talk_id: int, title: str) -> str:
        """Retitle a talk in flight. Returns the title as stored."""
        clean = title.strip()[:200] or "untitled"
        await asyncio.to_thread(self._rename, talk_id, clean)
        return clean

    async def update_usage(self, talk_id: int, usage: dict) -> None:
        await asyncio.to_thread(self._update_usage, talk_id, usage)

    async def end_talk(self, talk_id: int, usage: dict | None = None) -> None:
        await asyncio.to_thread(self._end, talk_id, usage)

    async def recent(self, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._recent, limit)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
