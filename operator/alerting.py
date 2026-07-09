"""
Pages a human. Uses a Slack incoming webhook if SLACK_WEBHOOK_URL is set
in the environment; otherwise falls back to printing a formatted alert to
stdout/logs so the whole pipeline still works with zero external config
(useful for the demo, CI, or air-gapped clusters).

Swap in PagerDuty/Opsgenie by adding another branch here — the interface
(`page(incident, summary)`) stays the same for the rest of the system.
"""

import logging
import os

logger = logging.getLogger("alerting")

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")


def page(incident, summary_text, action_result=None):
    payload_text = _format_alert(incident, summary_text, action_result)

    if SLACK_WEBHOOK_URL:
        try:
            import requests
            requests.post(SLACK_WEBHOOK_URL, json={"text": payload_text}, timeout=5)
            logger.info("Paged via Slack webhook.")
            return {"channel": "slack", "status": "sent"}
        except Exception as e:
            logger.error("Slack webhook failed (%s), falling back to stdout.", e)

    print("\n" + "=" * 70)
    print("🚨 PAGE (no SLACK_WEBHOOK_URL configured — printing instead) 🚨")
    print("=" * 70)
    print(payload_text)
    print("=" * 70 + "\n")
    return {"channel": "stdout", "status": "printed"}


def _format_alert(incident, summary_text, action_result):
    lines = [
        f"Severity: {incident['severity'].upper()}",
        f"Pod: {incident['pod']}",
        f"Signal: {incident['signal']}",
        f"Log lines: {incident['start_line']}-{incident['end_line']}",
        f"Anomaly score: {incident['worst_score']:.4f}",
        "",
        summary_text,
    ]
    if action_result:
        lines += ["", f"Automated action taken: {action_result}"]
    return "\n".join(lines)
