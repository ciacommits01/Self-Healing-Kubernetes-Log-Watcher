"""
Scores a (live or historical) log stream against the trained detector and
groups consecutive anomalous windows for the same pod into a single
"incident" — this is what gets handed to the operator + LLM layers.
"""

import sys

import joblib
import numpy as np

sys.path.append(".")
from features import extract_windows


class AnomalyDetector:
    def __init__(self, model_path="model.joblib"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        self.window_size = bundle["window_size"]
        self.stride = bundle["stride"]
        self.threshold = bundle["threshold"]
        self.critical_threshold = bundle.get("critical_threshold", bundle["threshold"] - 0.05)

    def score_lines(self, lines):
        windows = extract_windows(lines, window_size=self.window_size, stride=self.stride)
        if not windows:
            return []

        X = np.array([w["features"] for w in windows])
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)

        for w, s in zip(windows, scores):
            w["score"] = float(s)
            w["is_anomaly"] = bool(s < self.threshold)
        return windows

    def group_incidents(self, scored_windows, lines, gap_tolerance=2, min_windows=1):
        """
        Collapses consecutive flagged windows (per pod) into incidents,
        and attaches the dominant signal (restart/oom/conn/panic) plus the
        raw log excerpt for the LLM summarizer.

        min_windows filters out single isolated blips (e.g. one slow
        request) so only sustained anomalous behavior becomes an incident
        that pages someone — this is the equivalent of "alert hysteresis"
        in traditional monitoring.
        """
        by_pod = {}
        for w in scored_windows:
            by_pod.setdefault(w["pod"], []).append(w)

        incidents = []
        for pod, windows in by_pod.items():
            windows.sort(key=lambda w: w["end_line"])
            current = []
            last_line = None

            def flush(buf):
                if len(buf) < min_windows:
                    return
                start_line = buf[0]["end_line"] - self.window_size + 1
                end_line = buf[-1]["end_line"]
                start_line = max(start_line, 0)
                # Only keep lines belonging to this pod so the excerpt shown
                # to the LLM isn't polluted by interleaved neighbor pods.
                tag = f"[{pod}]"
                excerpt = [
                    l.rstrip("\n") for l in lines[start_line:end_line + 1] if tag in l
                ]
                worst_score = min(b["score"] for b in buf)
                signal = _dominant_signal(buf)
                severity = "critical" if worst_score < self.critical_threshold else "warning"
                incidents.append({
                    "pod": pod,
                    "start_line": start_line,
                    "end_line": end_line,
                    "worst_score": worst_score,
                    "severity": severity,
                    "signal": signal,
                    "excerpt": excerpt,
                })

            for w in windows:
                if not w["is_anomaly"]:
                    continue
                if last_line is not None and (w["end_line"] - last_line) > gap_tolerance * self.stride:
                    flush(current)
                    current = []
                current.append(w)
                last_line = w["end_line"]
            flush(current)

        incidents.sort(key=lambda inc: inc["start_line"])
        return self._merge_adjacent(incidents)

    def _merge_adjacent(self, incidents, merge_gap=None):
        """
        A single real-world incident (e.g. one crash loop) can get split
        into two adjacent groups if one window in the middle happened to
        score just above threshold. Without this, that shows up as two
        separate pages for the same event. We merge same-pod groups whose
        gap is within one window's worth of lines.
        """
        if merge_gap is None:
            merge_gap = self.window_size

        by_pod = {}
        for inc in incidents:
            by_pod.setdefault(inc["pod"], []).append(inc)

        merged = []
        for pod, group in by_pod.items():
            group.sort(key=lambda inc: inc["start_line"])
            current = group[0]
            for nxt in group[1:]:
                if nxt["start_line"] - current["end_line"] <= merge_gap:
                    current = {
                        "pod": pod,
                        "start_line": current["start_line"],
                        "end_line": nxt["end_line"],
                        "worst_score": min(current["worst_score"], nxt["worst_score"]),
                        "severity": "critical" if min(current["worst_score"], nxt["worst_score"]) < self.critical_threshold else "warning",
                        "signal": current["signal"] if current["worst_score"] <= nxt["worst_score"] else nxt["signal"],
                        "excerpt": current["excerpt"] + nxt["excerpt"],
                    }
                else:
                    merged.append(current)
                    current = nxt
            merged.append(current)

        merged.sort(key=lambda inc: inc["start_line"])
        return merged


_FEATURE_INDEX = {
    "log_rate": 0, "error_ratio": 1, "warn_ratio": 2, "restart_signal": 3,
    "oom_signal": 4, "conn_signal": 5, "panic_signal": 6,
    "distinct_error_types": 7, "avg_latency": 8, "latency_spike": 9,
}

_SIGNAL_LABELS = {
    "restart_signal": "crash_loop",
    "oom_signal": "oom_kill",
    "conn_signal": "dependency_timeout",
    "panic_signal": "crash_loop",
    "error_ratio": "error_spike",
}


def _dominant_signal(buf):
    """Averages the raw feature vectors across a flagged window group and
    picks whichever incident-signature feature is strongest, so the
    downstream summary can say *what kind* of incident this looks like."""
    feats = np.array([b["features"] for b in buf])
    avg = feats.mean(axis=0)

    candidates = {name: avg[_FEATURE_INDEX[name]] for name in _SIGNAL_LABELS}
    best_feature = max(candidates, key=candidates.get)
    if candidates[best_feature] <= 0:
        return "error_pattern"
    return _SIGNAL_LABELS[best_feature]


if __name__ == "__main__":
    detector = AnomalyDetector("model.joblib")
    with open("../data/live_stream.log") as f:
        lines = f.readlines()

    scored = detector.score_lines(lines)
    incidents = detector.group_incidents(scored, lines)

    print(f"Scored {len(scored)} windows, found {len(incidents)} incident(s):\n")
    for inc in incidents:
        print(f"[{inc['severity'].upper()}] Pod: {inc['pod']}  lines {inc['start_line']}-{inc['end_line']}  "
              f"signal={inc['signal']}  worst_score={inc['worst_score']:.4f}")
        if inc["excerpt"]:
            print("  Sample:", inc["excerpt"][0][:100])
        print()
