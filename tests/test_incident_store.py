"""
Unit tests for storage/incident_store.py

Uses a temporary SQLite database for each test so tests are hermetic.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "storage"))

from incident_store import init_db, save_incident, get_recent_incidents, get_stats, clear_all


def _fake_incident(**overrides):
    inc = {
        "pod": "test-pod-abc",
        "signal": "crash_loop",
        "severity": "critical",
        "worst_score": -0.75,
        "start_line": 10,
        "end_line": 30,
        "excerpt": ["line1", "line2"],
    }
    inc.update(overrides)
    return inc


class TestIncidentStore(unittest.TestCase):
    def setUp(self):
        # Each test gets its own isolated temp DB
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db = self._tmpfile.name
        self._tmpfile.close()
        init_db(self._db)

    def tearDown(self):
        os.unlink(self._db)

    # ---- save_incident ----

    def test_save_and_retrieve(self):
        save_incident(_fake_incident(), db_path=self._db)
        rows = get_recent_incidents(db_path=self._db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pod"], "test-pod-abc")
        self.assertEqual(rows[0]["signal"], "crash_loop")

    def test_save_with_action_and_summary(self):
        action = {"action": "restart_pod", "status": "dry_run"}
        save_incident(
            _fake_incident(), action_result=action,
            summary_text="Test summary.", backend="fallback",
            db_path=self._db,
        )
        rows = get_recent_incidents(db_path=self._db)
        self.assertIn("restart_pod", rows[0]["action_json"])
        self.assertEqual(rows[0]["summary"], "Test summary.")
        self.assertEqual(rows[0]["llm_backend"], "fallback")

    def test_save_warning_no_action(self):
        save_incident(_fake_incident(severity="warning"), db_path=self._db)
        rows = get_recent_incidents(db_path=self._db)
        self.assertEqual(rows[0]["severity"], "warning")
        self.assertIsNone(rows[0]["action_json"])

    def test_multiple_incidents_ordered_desc(self):
        save_incident(_fake_incident(pod="pod-1"), db_path=self._db)
        save_incident(_fake_incident(pod="pod-2"), db_path=self._db)
        rows = get_recent_incidents(db_path=self._db)
        # Most recent first (id DESC)
        self.assertEqual(rows[0]["pod"], "pod-2")
        self.assertEqual(rows[1]["pod"], "pod-1")

    def test_limit(self):
        for i in range(5):
            save_incident(_fake_incident(pod=f"pod-{i}"), db_path=self._db)
        rows = get_recent_incidents(limit=3, db_path=self._db)
        self.assertEqual(len(rows), 3)

    def test_severity_filter(self):
        save_incident(_fake_incident(severity="critical"), db_path=self._db)
        save_incident(_fake_incident(severity="warning"), db_path=self._db)
        crit = get_recent_incidents(severity="critical", db_path=self._db)
        warn = get_recent_incidents(severity="warning", db_path=self._db)
        self.assertEqual(len(crit), 1)
        self.assertEqual(crit[0]["severity"], "critical")
        self.assertEqual(len(warn), 1)

    # ---- get_stats ----

    def test_stats_empty(self):
        stats = get_stats(db_path=self._db)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["critical"], 0)
        self.assertEqual(stats["warning"], 0)
        self.assertEqual(stats["by_signal"], [])
        self.assertEqual(stats["by_pod"], [])

    def test_stats_counts(self):
        save_incident(_fake_incident(severity="critical", signal="crash_loop"), db_path=self._db)
        save_incident(_fake_incident(severity="critical", signal="oom_kill"), db_path=self._db)
        save_incident(_fake_incident(severity="warning", signal="error_spike"), db_path=self._db)
        stats = get_stats(db_path=self._db)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["critical"], 2)
        self.assertEqual(stats["warning"], 1)
        signal_names = [r["signal"] for r in stats["by_signal"]]
        self.assertIn("crash_loop", signal_names)

    def test_stats_by_pod(self):
        save_incident(_fake_incident(pod="pod-a"), db_path=self._db)
        save_incident(_fake_incident(pod="pod-a"), db_path=self._db)
        save_incident(_fake_incident(pod="pod-b"), db_path=self._db)
        stats = get_stats(db_path=self._db)
        by_pod = {r["pod"]: r["c"] for r in stats["by_pod"]}
        self.assertEqual(by_pod["pod-a"], 2)
        self.assertEqual(by_pod["pod-b"], 1)

    # ---- clear_all ----

    def test_clear_all(self):
        save_incident(_fake_incident(), db_path=self._db)
        save_incident(_fake_incident(), db_path=self._db)
        clear_all(db_path=self._db)
        rows = get_recent_incidents(db_path=self._db)
        self.assertEqual(len(rows), 0)
        stats = get_stats(db_path=self._db)
        self.assertEqual(stats["total"], 0)

    def test_idempotent_init(self):
        """Calling init_db twice should not raise or duplicate schema."""
        init_db(self._db)
        init_db(self._db)
        save_incident(_fake_incident(), db_path=self._db)
        rows = get_recent_incidents(db_path=self._db)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
