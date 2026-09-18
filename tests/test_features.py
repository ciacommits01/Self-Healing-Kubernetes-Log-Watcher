"""
Unit tests for detector/features.py

Covers:
  - parse_line: structured text, JSON, blank/unparsable lines
  - _keyword_hits
  - PodWindow: add, is_full, to_features edge cases
  - extract_windows: stride, unparsed-line reporting
"""

import sys
import os
import unittest

# Allow importing from detector/ regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector"))

from features import (
    parse_line,
    _keyword_hits,
    PodWindow,
    extract_windows,
    FEATURE_NAMES,
)


class TestParseLine(unittest.TestCase):
    # ---- structured text format ----

    def test_structured_info(self):
        line = "2026-01-01T00:00:00Z INFO  [my-pod-abc] hello world"
        r = parse_line(line)
        self.assertIsNotNone(r)
        self.assertEqual(r["level"], "INFO")
        self.assertEqual(r["pod"], "my-pod-abc")
        self.assertEqual(r["msg"], "hello world")

    def test_structured_error(self):
        line = "2026-01-01T00:00:01Z ERROR [payment-api-abc] connection refused"
        r = parse_line(line)
        self.assertIsNotNone(r)
        self.assertEqual(r["level"], "ERROR")

    def test_structured_warn(self):
        line = "2026-01-01T00:00:02Z WARN  [pod-x] memory usage at 90%"
        r = parse_line(line)
        self.assertIsNotNone(r)
        self.assertEqual(r["level"], "WARN")

    def test_structured_preserves_msg(self):
        line = "2026-01-01T00:00:03Z DEBUG [pod-x] Handled GET /health 200 12ms"
        r = parse_line(line)
        self.assertIsNotNone(r)
        self.assertIn("12ms", r["msg"])

    # ---- JSON format ----

    def test_json_standard_keys(self):
        import json
        obj = {"ts": "2026-01-01T00:00:00Z", "level": "ERROR",
               "pod": "api-pod", "message": "disk full"}
        r = parse_line(json.dumps(obj))
        self.assertIsNotNone(r)
        self.assertEqual(r["level"], "ERROR")
        self.assertEqual(r["pod"], "api-pod")
        self.assertEqual(r["msg"], "disk full")

    def test_json_alias_keys(self):
        import json
        obj = {"timestamp": "2026-01-01T00:00:00Z", "severity": "WARN",
               "pod_name": "worker-1", "text": "slow query"}
        r = parse_line(json.dumps(obj))
        self.assertIsNotNone(r)
        self.assertEqual(r["level"], "WARN")
        self.assertEqual(r["msg"], "slow query")

    def test_json_missing_pod_defaults(self):
        import json
        obj = {"ts": "2026-01-01T00:00:00Z", "level": "INFO", "message": "ok"}
        r = parse_line(json.dumps(obj))
        self.assertIsNotNone(r)
        self.assertEqual(r["pod"], "unknown")

    # ---- edge cases ----

    def test_blank_line(self):
        self.assertIsNone(parse_line(""))
        self.assertIsNone(parse_line("   \n"))

    def test_unparsable_returns_none(self):
        self.assertIsNone(parse_line("this is not a log line"))
        self.assertIsNone(parse_line("---"))

    def test_invalid_json(self):
        self.assertIsNone(parse_line("{broken json"))


class TestKeywordHits(unittest.TestCase):
    def test_single_hit(self):
        self.assertEqual(_keyword_hits("connection refused: dial", ["connection refused"]), 1)

    def test_no_hit(self):
        self.assertEqual(_keyword_hits("all good", ["timeout", "oom"]), 0)

    def test_multiple_hits(self):
        msg = "panic: nil pointer and traceback follows"
        self.assertEqual(_keyword_hits(msg, ["panic", "traceback", "timeout"]), 2)

    def test_empty_keywords(self):
        self.assertEqual(_keyword_hits("error", []), 0)


class TestPodWindow(unittest.TestCase):
    def _make_entry(self, level="INFO", msg="hello", ts="2026-01-01T00:00:00Z", pod="p"):
        return {"ts": ts, "level": level, "msg": msg, "pod": pod}

    def test_not_full_until_size(self):
        pw = PodWindow(size=3)
        self.assertFalse(pw.is_full())
        pw.add(self._make_entry())
        pw.add(self._make_entry())
        self.assertFalse(pw.is_full())
        pw.add(self._make_entry())
        self.assertTrue(pw.is_full())

    def test_to_features_length(self):
        pw = PodWindow(size=5)
        for _ in range(5):
            pw.add(self._make_entry())
        feats = pw.to_features()
        self.assertEqual(len(feats), len(FEATURE_NAMES))

    def test_to_features_empty(self):
        pw = PodWindow(size=5)
        feats = pw.to_features()
        self.assertEqual(feats, [0.0] * len(FEATURE_NAMES))

    def test_error_ratio_computed(self):
        pw = PodWindow(size=4)
        pw.add(self._make_entry(level="ERROR"))
        pw.add(self._make_entry(level="ERROR"))
        pw.add(self._make_entry(level="INFO"))
        pw.add(self._make_entry(level="INFO"))
        feats = pw.to_features()
        error_ratio_idx = FEATURE_NAMES.index("error_ratio")
        self.assertAlmostEqual(feats[error_ratio_idx], 0.5)

    def test_latency_extracted(self):
        pw = PodWindow(size=2)
        pw.add(self._make_entry(msg="Handled GET /health 200 150ms"))
        pw.add(self._make_entry(msg="Handled GET /health 200 50ms"))
        feats = pw.to_features()
        avg_lat_idx = FEATURE_NAMES.index("avg_latency")
        spike_idx = FEATURE_NAMES.index("latency_spike")
        self.assertAlmostEqual(feats[avg_lat_idx], 100.0)
        self.assertAlmostEqual(feats[spike_idx], 150.0)

    def test_oom_signal(self):
        pw = PodWindow(size=2)
        pw.add(self._make_entry(msg="oomkilled container"))
        pw.add(self._make_entry(msg="normal line"))
        feats = pw.to_features()
        oom_idx = FEATURE_NAMES.index("oom_signal")
        self.assertGreater(feats[oom_idx], 0.0)

    def test_novel_template_ratio(self):
        pw = PodWindow(size=4)
        pw.add({**self._make_entry(), "is_novel_template": True})
        pw.add({**self._make_entry(), "is_novel_template": True})
        pw.add(self._make_entry())
        pw.add(self._make_entry())
        feats = pw.to_features()
        novel_idx = FEATURE_NAMES.index("novel_template_ratio")
        self.assertAlmostEqual(feats[novel_idx], 0.5)

    def test_rolling_eviction(self):
        """deque maxlen evicts oldest entries correctly."""
        pw = PodWindow(size=2)
        pw.add(self._make_entry(level="ERROR"))
        pw.add(self._make_entry(level="ERROR"))
        pw.add(self._make_entry(level="INFO"))  # evicts first ERROR
        feats = pw.to_features()
        error_idx = FEATURE_NAMES.index("error_ratio")
        # 1 ERROR out of 2 remaining entries
        self.assertAlmostEqual(feats[error_idx], 0.5)


class TestExtractWindows(unittest.TestCase):
    def _make_line(self, pod="pod-a", level="INFO", msg="ok",
                   ts="2026-01-01T00:00:00Z"):
        return f"{ts} {level:<5} [{pod}] {msg}\n"

    def test_returns_windows_for_full_pod(self):
        lines = [self._make_line() for _ in range(25)]
        results = extract_windows(lines, window_size=5, stride=5)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("pod", r)
            self.assertIn("features", r)
            self.assertEqual(len(r["features"]), len(FEATURE_NAMES))

    def test_separate_windows_per_pod(self):
        lines = (
            [self._make_line(pod="pod-a") for _ in range(10)]
            + [self._make_line(pod="pod-b") for _ in range(10)]
        )
        results = extract_windows(lines, window_size=5, stride=5)
        pods = {r["pod"] for r in results}
        self.assertIn("pod-a", pods)
        self.assertIn("pod-b", pods)

    def test_blank_lines_ignored(self):
        lines = [self._make_line() for _ in range(10)] + ["", "   \n"]
        # Should not raise
        results = extract_windows(lines, window_size=5, stride=5)
        self.assertIsInstance(results, list)

    def test_short_stream_no_windows(self):
        """If there aren't enough lines to fill a window, no results."""
        lines = [self._make_line() for _ in range(3)]
        results = extract_windows(lines, window_size=10, stride=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
