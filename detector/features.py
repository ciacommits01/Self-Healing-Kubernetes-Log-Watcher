"""
Turns raw log lines into numeric feature vectors the model can learn from.

Design choice: rather than embedding raw text (which needs a pretrained
language model), we hand-engineer features that any SRE would recognize as
crash-loop / incident signals. This keeps the whole detector dependency-light,
fast to train, fully offline, and easy to explain in a report:

  - log_rate            lines seen per second in the window
  - error_ratio          fraction of lines at ERROR level
  - warn_ratio           fraction of lines at WARN level
  - restart_signal       hits for "restart", "back-off", "terminated", "starting container"
  - oom_signal           hits for "oomkilled", "memory usage"
  - conn_signal          hits for "connection refused", "timeout", "deadline exceeded"
  - panic_signal         hits for "panic", "exception", "traceback", "unhandled"
  - distinct_error_types fraction of unique error message templates in window
  - avg_latency          mean latency (ms) parsed out of request log lines
  - latency_spike        max latency in window relative to a fixed baseline

Windows are built PER POD over a sliding count of N consecutive lines
(default 20), which is more robust for bursty log traffic than fixed-time
windows.
"""

import re
from collections import defaultdict, deque

LOG_RE = re.compile(
    r"^(?P<ts>\S+)\s+(?P<level>\w+)\s+\[(?P<pod>[^\]]+)\]\s+(?P<msg>.*)$"
)
LATENCY_RE = re.compile(r"(\d+)ms")

RESTART_KEYWORDS = ["back-off", "restarting", "terminated", "starting container"]
OOM_KEYWORDS = ["oomkilled", "memory usage"]
CONN_KEYWORDS = ["connection refused", "timeout", "deadline exceeded", "503"]
PANIC_KEYWORDS = ["panic", "exception", "traceback", "unhandled", "nil pointer"]

FEATURE_NAMES = [
    "log_rate",
    "error_ratio",
    "warn_ratio",
    "restart_signal",
    "oom_signal",
    "conn_signal",
    "panic_signal",
    "distinct_error_types",
    "avg_latency",
    "latency_spike",
]


def parse_line(line):
    m = LOG_RE.match(line.strip())
    if not m:
        return None
    return m.groupdict()


def _keyword_hits(msg_lower, keywords):
    return sum(1 for k in keywords if k in msg_lower)


class PodWindow:
    """Rolling window of parsed log entries for a single pod."""

    def __init__(self, size=20):
        self.size = size
        self.entries = deque(maxlen=size)

    def add(self, entry):
        self.entries.append(entry)

    def is_full(self):
        return len(self.entries) == self.size

    def to_features(self):
        entries = list(self.entries)
        n = len(entries)
        if n == 0:
            return [0.0] * len(FEATURE_NAMES)

        error_count = 0
        warn_count = 0
        restart_hits = 0
        oom_hits = 0
        conn_hits = 0
        panic_hits = 0
        error_messages = set()
        latencies = []

        t0 = entries[0]["ts"]
        t1 = entries[-1]["ts"]

        for e in entries:
            level = e["level"].upper()
            msg_lower = e["msg"].lower()

            if level == "ERROR":
                error_count += 1
                error_messages.add(msg_lower[:40])
            elif level == "WARN":
                warn_count += 1

            restart_hits += _keyword_hits(msg_lower, RESTART_KEYWORDS)
            oom_hits += _keyword_hits(msg_lower, OOM_KEYWORDS)
            conn_hits += _keyword_hits(msg_lower, CONN_KEYWORDS)
            panic_hits += _keyword_hits(msg_lower, PANIC_KEYWORDS)

            lat = LATENCY_RE.search(msg_lower)
            if lat:
                latencies.append(int(lat.group(1)))

        try:
            from datetime import datetime

            dt0 = datetime.fromisoformat(t0.replace("Z", "+00:00"))
            dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
            duration = max((dt1 - dt0).total_seconds(), 0.01)
        except Exception:
            duration = 1.0

        log_rate = n / duration
        error_ratio = error_count / n
        warn_ratio = warn_count / n
        distinct_error_types = (len(error_messages) / error_count) if error_count else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        latency_spike = max(latencies) if latencies else 0.0

        return [
            log_rate,
            error_ratio,
            warn_ratio,
            restart_hits / n,
            oom_hits / n,
            conn_hits / n,
            panic_hits / n,
            distinct_error_types,
            avg_latency,
            latency_spike,
        ]


def extract_windows(lines, window_size=20, stride=5):
    """
    Streams through log lines, grouping per-pod, and yields
    (pod, end_line_index, feature_vector) for every full window.
    Stride controls how often we emit a window (5 = emit every 5 new lines).
    """
    pod_windows = defaultdict(lambda: PodWindow(size=window_size))
    pod_counters = defaultdict(int)
    results = []

    for i, raw in enumerate(lines):
        parsed = parse_line(raw)
        if not parsed:
            continue
        pod = parsed["pod"]
        pod_windows[pod].add(parsed)
        pod_counters[pod] += 1

        if pod_windows[pod].is_full() and pod_counters[pod] % stride == 0:
            feats = pod_windows[pod].to_features()
            results.append({"pod": pod, "end_line": i, "features": feats})

    return results
