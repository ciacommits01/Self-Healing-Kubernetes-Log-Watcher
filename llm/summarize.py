"""
Turns a detected incident (pod, signal, score, raw log excerpt) into a
human-readable incident summary — the kind an SRE would want in a Slack
alert or postmortem doc.

LLM backend: Ollama (https://ollama.com), which is free, open-weight, and
runs entirely on your own machine/cluster (no API key, no per-token cost).
    1. Install: https://ollama.com/download
    2. Pull a small model:  ollama pull llama3.2:1b   (or phi3, mistral, etc.)
    3. It serves an OpenAI-compatible-ish REST API at localhost:11434.

If Ollama isn't running (e.g. in CI, or you haven't installed it yet), we
fall back to a deterministic template summary built from the same fields —
this keeps the whole pipeline runnable out of the box with zero setup, and
means a flaky/absent LLM never blocks a page from going out.
"""

import json
import logging

import requests

logger = logging.getLogger("summarize")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # swap for any model you've pulled locally

PROMPT_TEMPLATE = """You are an SRE assistant writing a concise incident summary for a Slack alert.

Incident data:
- Pod: {pod}
- Detected signal: {signal}
- Anomaly severity: {severity} (isolation-forest score: {score:.4f}, more negative = more anomalous)
- Log line range: {start_line}-{end_line}
- Automated action taken: {action}

Raw log excerpt:
{excerpt}

Write a 3-4 sentence incident summary for an on-call engineer. State what
happened, the likely root cause based on the log content, what automated
action was already taken, and one concrete next step for the human. Be
direct and specific, no filler, no markdown headers."""


def _call_ollama(prompt, timeout=15):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _fallback_summary(incident, action_result):
    """Deterministic, template-based summary — no LLM required. Used when
    Ollama isn't reachable, so the pipeline degrades gracefully instead of
    silently dropping the alert."""
    signal_readable = incident["signal"].replace("_", " ")
    action_desc = "no automated action was taken"
    if action_result:
        a = action_result.get("action", "")
        status = action_result.get("status", "")
        if a == "restart_pod":
            action_desc = f"the pod was {'restarted' if status == 'executed' else 'flagged for restart (dry-run)'}"
        elif a == "scale_deployment":
            action_desc = f"the deployment was {'scaled up' if status == 'executed' else 'flagged to scale up (dry-run)'}"

    sample_lines = incident["excerpt"][:3]
    sample = " | ".join(l.split("] ", 1)[-1] for l in sample_lines) if sample_lines else "n/a"

    return (
        f"{incident['severity'].upper()} incident on pod {incident['pod']}: detected a "
        f"{signal_readable} pattern (anomaly score {incident['worst_score']:.3f}) across "
        f"log lines {incident['start_line']}-{incident['end_line']}. Representative log "
        f"content: {sample}. In response, {action_desc}. Recommended next step: review "
        f"the pod's recent deploy history and dependency health, then confirm the "
        f"automated action resolved the issue before closing this alert."
    )


def summarize_incident(incident, action_result=None, use_llm=True):
    """
    Returns (summary_text, backend) where backend is 'ollama' or 'fallback'
    so callers/logs can tell which path was used.
    """
    if use_llm:
        prompt = PROMPT_TEMPLATE.format(
            pod=incident["pod"],
            signal=incident["signal"],
            severity=incident["severity"],
            score=incident["worst_score"],
            start_line=incident["start_line"],
            end_line=incident["end_line"],
            action=json.dumps(action_result) if action_result else "none",
            excerpt="\n".join(incident["excerpt"][:15]),
        )
        try:
            text = _call_ollama(prompt)
            return text, "ollama"
        except Exception as e:
            logger.warning("Ollama unavailable (%s) — using template fallback.", e)

    return _fallback_summary(incident, action_result), "fallback"


if __name__ == "__main__":
    # Quick manual test with a fake incident
    fake_incident = {
        "pod": "payment-api-7d9f8b6c-x2kqp",
        "signal": "oom_kill",
        "severity": "critical",
        "worst_score": -0.7435,
        "start_line": 1512,
        "end_line": 1521,
        "excerpt": [
            "2026-07-06T09:10:01.288Z WARN  [payment-api-7d9f8b6c-x2kqp] Memory usage at 92% of limit",
            "2026-07-06T09:10:01.882Z ERROR [payment-api-7d9f8b6c-x2kqp] Container payment-api killed: OOMKilled",
            "2026-07-06T09:10:02.437Z INFO  [payment-api-7d9f8b6c-x2kqp] Starting container payment-api",
        ],
    }
    fake_action = {"action": "restart_pod", "pod": fake_incident["pod"], "status": "dry_run"}
    text, backend = summarize_incident(fake_incident, fake_action)
    print(f"[backend={backend}]\n{text}")
