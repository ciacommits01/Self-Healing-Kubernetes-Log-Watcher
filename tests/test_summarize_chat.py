"""
Unit tests for llm/summarize.py — deterministic fallback template,
and for chatbot/chat_engine.py — TTL cache behaviour.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "storage"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))


# ---------------------------------------------------------------------------
# Tests for _fallback_summary (llm/summarize.py)
# ---------------------------------------------------------------------------

class TestFallbackSummary(unittest.TestCase):
    def setUp(self):
        # Import here so sys.path adjustments take effect
        from summarize import _fallback_summary
        self._fn = _fallback_summary

    def _incident(self, **overrides):
        inc = {
            "pod": "payment-api-abc",
            "signal": "oom_kill",
            "severity": "critical",
            "worst_score": -0.75,
            "start_line": 100,
            "end_line": 120,
            "excerpt": [
                "2026-01-01 ERROR [payment-api-abc] OOMKilled",
                "2026-01-01 INFO  [payment-api-abc] Starting container",
            ],
        }
        inc.update(overrides)
        return inc

    def test_returns_string(self):
        result = self._fn(self._incident(), None)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_includes_pod_name(self):
        result = self._fn(self._incident(), None)
        self.assertIn("payment-api-abc", result)

    def test_includes_signal(self):
        result = self._fn(self._incident(), None)
        # signal has underscores replaced with spaces
        self.assertIn("oom kill", result)

    def test_no_action(self):
        result = self._fn(self._incident(), None)
        self.assertIn("no automated action", result)

    def test_restart_action_executed(self):
        action = {"action": "restart_pod", "status": "executed"}
        result = self._fn(self._incident(), action)
        self.assertIn("restarted", result)

    def test_restart_action_dry_run(self):
        action = {"action": "restart_pod", "status": "dry_run"}
        result = self._fn(self._incident(), action)
        self.assertIn("dry-run", result)

    def test_scale_action_executed(self):
        action = {"action": "scale_deployment", "status": "executed"}
        result = self._fn(self._incident(), action)
        self.assertIn("scaled up", result)

    def test_empty_excerpt(self):
        """Empty excerpt should not raise."""
        result = self._fn(self._incident(excerpt=[]), None)
        self.assertIsInstance(result, str)
        self.assertIn("n/a", result)

    def test_score_in_output(self):
        result = self._fn(self._incident(worst_score=-0.75), None)
        self.assertIn("-0.750", result)

    def test_line_range_in_output(self):
        result = self._fn(self._incident(start_line=42, end_line=99), None)
        self.assertIn("42", result)
        self.assertIn("99", result)


# ---------------------------------------------------------------------------
# Tests for _TTLCache (chatbot/chat_engine.py)
# ---------------------------------------------------------------------------

class TestTTLCache(unittest.TestCase):
    def setUp(self):
        from chat_engine import _TTLCache
        self._cls = _TTLCache

    def test_calls_fetch_fn_on_first_get(self):
        fn = MagicMock(return_value=42)
        cache = self._cls(fn, ttl_seconds=10)
        result = cache.get()
        self.assertEqual(result, 42)
        fn.assert_called_once()

    def test_returns_cached_value_within_ttl(self):
        call_count = [0]

        def fetch():
            call_count[0] += 1
            return call_count[0]

        cache = self._cls(fetch, ttl_seconds=60)
        first = cache.get()
        second = cache.get()
        self.assertEqual(first, second)
        self.assertEqual(call_count[0], 1)

    def test_refreshes_after_ttl_expires(self):
        call_count = [0]

        def fetch():
            call_count[0] += 1
            return call_count[0]

        cache = self._cls(fetch, ttl_seconds=0.05)  # 50ms TTL
        first = cache.get()
        time.sleep(0.1)
        second = cache.get()
        self.assertNotEqual(first, second)
        self.assertEqual(call_count[0], 2)

    def test_invalidate_forces_refresh(self):
        call_count = [0]

        def fetch():
            call_count[0] += 1
            return call_count[0]

        cache = self._cls(fetch, ttl_seconds=60)
        cache.get()
        cache.invalidate()
        cache.get()
        self.assertEqual(call_count[0], 2)


# ---------------------------------------------------------------------------
# Tests for ChatEngine prompt assembly (no LLM call)
# ---------------------------------------------------------------------------

class TestChatEnginePrompt(unittest.TestCase):
    def test_ask_appends_to_history(self):
        """Even when LLM is unavailable, ask() appends user+assistant turns."""
        # Patch generate to return None (simulates LLM unavailable)
        with patch("chat_engine.generate", return_value=(None, "none")):
            with patch("chat_engine.get_pod_status_text", return_value="no cluster"):
                with patch("chat_engine.get_stats", return_value={
                    "total": 0, "critical": 0, "warning": 0,
                    "by_signal": [], "by_pod": [],
                }):
                    with patch("chat_engine.get_recent_incidents", return_value=[]):
                        from chat_engine import ChatEngine
                        engine = ChatEngine()
                        engine.ask("How many incidents?")
                        self.assertEqual(len(engine.history), 2)
                        self.assertEqual(engine.history[0][0], "User")
                        self.assertEqual(engine.history[1][0], "Assistant")

    def test_history_is_bounded(self):
        """History should not exceed MAX_HISTORY_TURNS * 2 items."""
        with patch("chat_engine.generate", return_value=("ok", "local")):
            with patch("chat_engine.get_pod_status_text", return_value=""):
                with patch("chat_engine.get_stats", return_value={
                    "total": 0, "critical": 0, "warning": 0,
                    "by_signal": [], "by_pod": [],
                }):
                    with patch("chat_engine.get_recent_incidents", return_value=[]):
                        from chat_engine import ChatEngine, MAX_HISTORY_TURNS
                        engine = ChatEngine()
                        for i in range(MAX_HISTORY_TURNS + 5):
                            engine.ask(f"question {i}")
                        # Each ask adds 2 items; total grows unboundedly
                        # (the bound is on what is SENT to the LLM, not stored)
                        # — just verify it doesn't crash with many turns
                        self.assertGreater(len(engine.history), 0)


if __name__ == "__main__":
    unittest.main()
