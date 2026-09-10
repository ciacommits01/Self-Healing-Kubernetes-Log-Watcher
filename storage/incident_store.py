"""
SQLite persistence layer for incident history.

This is the foundational piece the dashboard and chatbot both read from.
Deliberately boring by design: one table, no ORM, no migrations framework
— a project this size doesn't need one, and adding one would be exactly
the kind of unnecessary complexity worth avoiding.

Every incident the detector groups (both WARNING and CRITICAL severity)
gets a row, so the dashboard/chatbot can show the full picture, not just
what got paged.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "incidents.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    pod TEXT NOT NULL,
    signal TEXT NOT NULL,
    severity TEXT NOT NULL,
    score REAL NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    excerpt TEXT,
    action_json TEXT,
    summary TEXT,
    llm_backend TEXT
);
CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
"""


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_incident(incident, action_result=None, summary_text=None, backend=None, db_path=DB_PATH):
    """Persists one incident row. Called for BOTH warning and critical
    incidents (action_result/summary_text/backend will be None for
    warning-tier ones, since those don't trigger action/summary/page)."""
    init_db(db_path)  # cheap no-op if already initialized; keeps this function standalone-safe
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO incidents
           (detected_at, pod, signal, severity, score, start_line, end_line,
            excerpt, action_json, summary, llm_backend)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            incident["pod"],
            incident["signal"],
            incident["severity"],
            incident["worst_score"],
            incident.get("start_line"),
            incident.get("end_line"),
            json.dumps(incident.get("excerpt", [])),
            json.dumps(action_result) if action_result else None,
            summary_text,
            backend,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_incidents(limit=20, severity=None, db_path=DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if severity:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE severity = ? ORDER BY id DESC LIMIT ?",
            (severity, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(db_path=DB_PATH):
    """Summary counts for the dashboard header and the chatbot's context."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as c FROM incidents").fetchone()["c"]
    critical = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE severity='critical'").fetchone()["c"]
    warning = conn.execute("SELECT COUNT(*) as c FROM incidents WHERE severity='warning'").fetchone()["c"]

    by_signal = conn.execute(
        "SELECT signal, COUNT(*) as c FROM incidents GROUP BY signal ORDER BY c DESC"
    ).fetchall()
    by_pod = conn.execute(
        "SELECT pod, COUNT(*) as c FROM incidents GROUP BY pod ORDER BY c DESC LIMIT 10"
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "critical": critical,
        "warning": warning,
        "by_signal": [dict(r) for r in by_signal],
        "by_pod": [dict(r) for r in by_pod],
    }


def clear_all(db_path=DB_PATH):
    """Wipes incident history — useful between demo runs. Not exposed via
    the dashboard UI on purpose (no destructive actions behind a button
    with no confirmation); call this directly if you want a clean slate."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM incidents")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Quick manual smoke test
    init_db()
    fake = {
        "pod": "payment-api-test",
        "signal": "oom_kill",
        "severity": "critical",
        "worst_score": -0.74,
        "start_line": 100,
        "end_line": 110,
        "excerpt": ["line1", "line2"],
    }
    save_incident(fake, action_result={"action": "restart_pod", "status": "dry_run"},
                  summary_text="Test summary.", backend="fallback")
    print("Stats:", get_stats())
    print("Recent:", get_recent_incidents(5))
