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
import sys

sys.path.append("detector")
sys.path.append("operator")
sys.path.append("llm")

from detect import AnomalyDetector  # noqa: E402
from features import parse_line  # noqa: E402
from watcher import SimulatedWatcher, LiveK8sWatcher  # noqa: E402
from k8s_actions import take_action  # noqa: E402
from alerting import page  # noqa: E402
from summarize import summarize_incident  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


class StreamingRunner:
    """
    Wraps AnomalyDetector's batch API (extract_windows over a full list of
    lines) into an incremental, line-at-a-time consumer suitable for a
    live tail. Buffers lines per pod and re-scores whenever a pod's window
    fills up, so we can react while the stream is still flowing rather
    than waiting for it to end.
    """

    def __init__(self, model_path, namespace="default"):
        self.detector = AnomalyDetector(model_path)
        self.namespace = namespace
        self.all_lines = []
        self.seen_incident_ranges = []  # list of (pod, start_line, end_line) already actioned

    def on_line(self, line):
        self.all_lines.append(line)
        if len(self.all_lines) % self.detector.stride != 0:
            return

        scored = self.detector.score_lines(self.all_lines)
        incidents = self.detector.group_incidents(scored, self.all_lines)

        for inc in incidents:
            if self._already_handled(inc):
                continue
            self.seen_incident_ranges.append((inc["pod"], inc["start_line"], inc["end_line"]))
            self._handle_incident(inc)

    def _already_handled(self, incident):
        for pod, start, end in self.seen_incident_ranges:
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
            return

        logger.warning(
            "CRITICAL incident: pod=%s signal=%s score=%.4f",
            incident["pod"], incident["signal"], incident["worst_score"],
        )

        action_result = take_action(incident["signal"], incident["pod"], self.namespace)
        summary_text, backend = summarize_incident(incident, action_result)
        logger.info("Summary generated via backend=%s", backend)
        page(incident, summary_text, action_result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    p.add_argument("--log-file", default="data/live_stream.log", help="for --mode simulate")
    p.add_argument("--model", default="detector/model.joblib")
    p.add_argument("--namespace", default="default")
    p.add_argument("--label-selector", default="app=payment-api", help="for --mode live")
    p.add_argument("--speed", type=float, default=0.0, help="seconds between lines in simulate mode")
    args = p.parse_args()

    runner = StreamingRunner(args.model, namespace=args.namespace)

    if args.mode == "simulate":
        logger.info("Running in SIMULATE mode against %s", args.log_file)
        watcher = SimulatedWatcher(args.log_file, speed=args.speed)
    else:
        logger.info("Running in LIVE mode: namespace=%s selector=%s", args.namespace, args.label_selector)
        watcher = LiveK8sWatcher(args.namespace, args.label_selector)

    watcher.run(runner.on_line)

    logger.info("Stream ended. Processed %d lines total.", len(runner.all_lines))


if __name__ == "__main__":
    main()
