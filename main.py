"""
Self-Healing Kubernetes Log Watcher — end-to-end orchestrator.

Pipeline:
    watcher (tail logs)
        -> rolling per-pod feature windows (detector/features.py)
        -> IsolationForest anomaly scoring (detector/detect.py)
        -> on CRITICAL incident:
             - take an automated remediation action (operator/k8s_actions.py)
             - generate a human-readable summary (llm/summarize.py)
             - page a human with both (operator/alerting.py)
        -> on WARNING incident: log it, no page, no action

Run modes:
    python main.py --mode simulate                 # demo, no cluster needed
    python main.py --mode live --namespace prod --label-selector app=payment-api

By default this runs with DRY_RUN=True in k8s_actions.py, so it will log
every action it *would* take without touching your cluster. Flip DRY_RUN
only once you trust the alerts you're seeing.
"""

import argparse
import logging
import threading
import sys
from collections import deque

sys.path.append("detector")
sys.path.append("operator")
sys.path.append("llm")
sys.path.append("storage")

from detect import AnomalyDetector  # noqa: E402
from features import parse_line  # noqa: E402
from watcher import SimulatedWatcher, LiveK8sWatcher  # noqa: E402
from k8s_actions import take_action, set_dry_run  # noqa: E402
from alerting import page  # noqa: E402
from summarize import summarize_incident  # noqa: E402
from incident_store import save_incident  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

# Maximum recent incident ranges to remember for dedup.  500 is far more than
# enough to cover any realistic burst of unique incidents before old ones age
# out naturally; keeping this bounded prevents memory growth on long runs.
_MAX_SEEN_RANGES = 500


class StreamingRunner:
    """
    Wraps AnomalyDetector's batch API into an incremental, line-at-a-time
    consumer suitable for a live tail.

    Key design decisions:

    * Uses a fixed-size sliding ring buffer (collections.deque with maxlen)
      instead of an unbounded list — memory stays O(buffer_size) regardless
      of stream length.  Previously, self.all_lines grew forever and was
      re-scored in full on every stride, causing O(N) CPU and unbounded RAM.

    * Maintains a monotonic _line_offset counter so that start_line/end_line
      values reported in incidents are global (absolute) line numbers, not
      relative positions within the buffer.  This keeps incident line ranges
      consistent across the lifetime of the process.

    * seen_incident_ranges is a bounded deque (maxlen=_MAX_SEEN_RANGES) so
      it cannot grow unbounded on a long-running stream.

    * All shared mutable state is accessed exclusively under self._lock so
      concurrent pod-log threads (live mode) cannot race each other.
    """

    def __init__(self, model_path, namespace="default", buffer_size=2000):
        self.detector = AnomalyDetector(model_path)
        self.namespace = namespace
        # Bounded ring buffer — old lines are automatically evicted when full.
        self._buffer = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        # Monotonic count of lines seen (never resets).
        self._line_offset = 0
        # Dedup tracker — bounded deque so it never grows unbounded.
        self._seen_incident_ranges = deque(maxlen=_MAX_SEEN_RANGES)
        self._lock = threading.Lock()

    @property
    def lines_processed(self):
        """Total lines consumed from the stream (thread-safe read)."""
        with self._lock:
            return self._line_offset

    def on_line(self, line):
        with self._lock:
            self._on_line_locked(line)

    def _on_line_locked(self, line):
        self._buffer.append(line)
        self._line_offset += 1

        if self._line_offset % self.detector.stride != 0:
            return

        # Snapshot the buffer as a plain list for a single scoring pass.
        buffer_snapshot = list(self._buffer)

        # The buffer may have fewer lines than buffer_size early in the stream.
        # buffer_start is the global line number of buffer_snapshot[0].
        buffer_start = self._line_offset - len(buffer_snapshot)

        scored = self.detector.score_lines(buffer_snapshot)
        incidents = self.detector.group_incidents(scored, buffer_snapshot)

        for inc in incidents:
            # Translate buffer-relative indices to global line numbers so
            # dashboards and alerts show meaningful, stable context.
            inc["start_line"] = buffer_start + inc.get("start_line", 0)
            inc["end_line"] = buffer_start + inc.get("end_line", 0)

            if self._already_handled(inc):
                continue
            self._seen_incident_ranges.append(
                (inc["pod"], inc["start_line"], inc["end_line"])
            )
            self._handle_incident(inc)

    def _already_handled(self, incident):
        for pod, start, end in self._seen_incident_ranges:
            if pod != incident["pod"]:
                continue
            # overlapping (or touching) ranges for the same pod = same incident
            if not (incident["end_line"] < start or incident["start_line"] > end):
                return True
        return False

    def _handle_incident(self, incident):
        if incident["severity"] == "warning":
            logger.info(
                "WARNING-tier anomaly (no action/page): pod=%s signal=%s score=%.4f",
                incident["pod"], incident["signal"], incident["worst_score"],
            )
            save_incident(incident)
            return

        logger.warning(
            "CRITICAL incident: pod=%s signal=%s score=%.4f",
            incident["pod"], incident["signal"], incident["worst_score"],
        )

        action_result = take_action(incident["signal"], incident["pod"], self.namespace)
        summary_text, backend = summarize_incident(incident, action_result)
        logger.info("Summary generated via backend=%s", backend)
        save_incident(incident, action_result, summary_text, backend)
        page(incident, summary_text, action_result)


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    p.add_argument("--log-file", default="data/live_stream.log", help="for --mode simulate")
    p.add_argument("--model", default="detector/model.joblib")
    p.add_argument("--namespace", default="default")
    p.add_argument("--label-selector", default="app=payment-api", help="for --mode live")
    p.add_argument("--speed", type=float, default=0.0,
                   help="seconds between lines in simulate mode")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                   help="Run actions in dry-run mode (default: True)")
    p.add_argument("--buffer-size", type=int, default=2000,
                   help="Max lines kept in the sliding window buffer (default: 2000). "
                        "Larger values improve incident grouping on very noisy streams.")
    args = p.parse_args()

    set_dry_run(args.dry_run)
    logger.info("Operating with DRY_RUN=%s", args.dry_run)

    runner = StreamingRunner(args.model, namespace=args.namespace,
                             buffer_size=args.buffer_size)

    if args.mode == "simulate":
        log_file = args.log_file
        if not os.path.exists(log_file) and os.path.exists("live_stream.log"):
            log_file = "live_stream.log"
        logger.info("Running in SIMULATE mode against %s", log_file)
        watcher = SimulatedWatcher(log_file, speed=args.speed)
    else:
        logger.info(
            "Running in LIVE mode: namespace=%s selector=%s",
            args.namespace, args.label_selector,
        )
        watcher = LiveK8sWatcher(args.namespace, args.label_selector)

    watcher.run(runner.on_line)

    logger.info("Stream ended. Processed %d lines total.", runner.lines_processed)


if __name__ == "__main__":
    main()
