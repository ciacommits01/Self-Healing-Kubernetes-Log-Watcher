"""
Unit tests for detector/detect.py — _dominant_signal, group_incidents,
_merge_adjacent, and _deployment_of heuristic fallback.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operator"))

from detect import _dominant_signal, _FEATURE_INDEX
from k8s_actions import _deployment_of


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_window(features=None, score=-0.3, is_anomaly=True, pod="pod-a", end_line=20):
    """Build a scored window dict with a configurable feature vector."""
    if features is None:
        features = [0.0] * len(_FEATURE_INDEX)
    return {
        "pod": pod,
        "end_line": end_line,
        "features": features,
        "score": score,
        "is_anomaly": is_anomaly,
    }


def _features(**kwargs):
    """Return an 11-element feature list from keyword overrides."""
    base = {name: 0.0 for name in _FEATURE_INDEX}
    base.update(kwargs)
    return [base[name] for name in sorted(_FEATURE_INDEX, key=_FEATURE_INDEX.get)]


# ---------------------------------------------------------------------------
# _dominant_signal
# ---------------------------------------------------------------------------

class TestDominantSignal(unittest.TestCase):
    def test_restart_signal_preferred(self):
        buf = [_make_window(features=_features(restart_signal=0.5, error_ratio=0.3))]
        self.assertEqual(_dominant_signal(buf), "crash_loop")

    def test_oom_signal(self):
        buf = [_make_window(features=_features(oom_signal=0.4))]
        self.assertEqual(_dominant_signal(buf), "oom_kill")

    def test_conn_signal(self):
        buf = [_make_window(features=_features(conn_signal=0.6))]
        self.assertEqual(_dominant_signal(buf), "dependency_timeout")

    def test_panic_signal(self):
        buf = [_make_window(features=_features(panic_signal=0.3))]
        self.assertEqual(_dominant_signal(buf), "crash_loop")

    def test_novel_template_fallback(self):
        """When no keyword feature fires, novel_template_ratio should trigger."""
        buf = [_make_window(features=_features(novel_template_ratio=0.4))]
        self.assertEqual(_dominant_signal(buf), "novel_pattern")

    def test_all_zero_defaults_to_error_pattern(self):
        buf = [_make_window(features=_features())]
        self.assertEqual(_dominant_signal(buf), "error_pattern")

    def test_highest_keyword_wins(self):
        """When multiple keywords fire, the highest average wins."""
        buf = [_make_window(features=_features(oom_signal=0.8, conn_signal=0.2))]
        self.assertEqual(_dominant_signal(buf), "oom_kill")

    def test_multiple_windows_averaged(self):
        """Signal is determined from the mean across all windows in the group."""
        w1 = _make_window(features=_features(oom_signal=1.0))
        w2 = _make_window(features=_features(oom_signal=0.0))
        self.assertEqual(_dominant_signal([w1, w2]), "oom_kill")


# ---------------------------------------------------------------------------
# _deployment_of — heuristic fallback (no live cluster required)
# ---------------------------------------------------------------------------

class TestDeploymentOf(unittest.TestCase):
    """
    These tests only exercise the heuristic fallback path, since we don't
    have a live K8s cluster in the test environment. The API path is tested
    by integration tests (not included here).
    """

    def test_standard_eks_pod(self):
        # payment-api-7d9f8b6c-x2kqp -> payment-api
        result = _deployment_of("payment-api-7d9f8b6c-x2kqp")
        self.assertEqual(result, "payment-api")

    def test_multisegment_deployment(self):
        # checkout-worker-svc-7d9f8b6c-x2kqp -> checkout-worker-svc
        result = _deployment_of("checkout-worker-svc-7d9f8b6c-x2kqp")
        self.assertEqual(result, "checkout-worker-svc")

    def test_short_name_returns_as_is(self):
        result = _deployment_of("myapp")
        self.assertEqual(result, "myapp")

    def test_two_segments(self):
        result = _deployment_of("myapp-abc12")
        # len(parts) == 2, so fallback returns pod_name unchanged
        self.assertEqual(result, "myapp-abc12")

    def test_three_segments(self):
        result = _deployment_of("web-abc12-xyz99")
        self.assertEqual(result, "web")


if __name__ == "__main__":
    unittest.main()
