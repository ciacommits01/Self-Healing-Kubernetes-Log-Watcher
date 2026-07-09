"""
Turns a detected incident (pod, signal, score, raw log excerpt) into a
human-readable incident summary — the kind an SRE would want in a Slack
alert or postmortem doc.

LLM backend: Ollama, in either of two modes — pick whichever fits:

  1. OLLAMA CLOUD (no local install): create an API key at
     https://ollama.com/settings/keys, then:
         export OLLAMA_API_KEY=your_api_key
     This calls https://ollama.com/api/generate with the key as a Bearer
     token. No GPU, no local model download, nothing running on your
     machine. Trade-off: your log excerpts leave your machine and go to
     Ollama's cloud service, and it's rate-limited/"free to start" rather
     than unconditionally free — check current limits on ollama.com before
     relying on it for a production pipeline.

  2. LOCAL OLLAMA (fully offline, no API key, no per-request cost):
         ollama pull llama3.2:1b   (or gpt-oss:20b, phi3, mistral, etc.)
         ollama serve
     Used automatically whenever OLLAMA_API_KEY is NOT set.

Priority: if OLLAMA_API_KEY is set, cloud is used. Otherwise it tries your
local Ollama at localhost:11434. If NEITHER is reachable (not installed,
no key, CI environment, etc.), we fall back to a deterministic template
summary built from the same incident fields — so a flaky/absent LLM never
silently blocks a page from going out.
"""

import json
import logging
import os

import requests

logger = logging.getLogger("summarize")

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_CLOUD_URL = "https://ollama.com/api/generate"
OLLAMA_CLOUD_MODEL = "gpt-oss:120b"  # only reachable via the cloud endpoint

OLLAMA_LOCAL_URL = "http://localhost:11434/api/generate"
OLLAMA_LOCAL_MODEL = "llama3.2:1b"  # swap for any model you've pulled locally

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


def _call_ollama_cloud(prompt, timeout=30):
    resp = requests.post(
        OLLAMA_CLOUD_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        json={"model": OLLAMA_CLOUD_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _call_ollama_local(prompt, timeout=15):
    resp = requests.post(
        OLLAMA_LOCAL_URL,
        json={"model": OLLAMA_LOCAL_MODEL, "prompt": prompt, "stream": False},
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
    Returns (summary_text, backend) where backend is 'ollama-cloud',
    'ollama-local', or 'fallback' so callers/logs can tell which path
    was actually used — useful since this silently degrades rather than
    erroring out.
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

        if OLLAMA_API_KEY:
            try:
                return _call_ollama_cloud(prompt), "ollama-cloud"
            except Exception as e:
                logger.warning("Ollama Cloud call failed (%s) — trying local Ollama next.", e)

        try:
            return _call_ollama_local(prompt), "ollama-local"
        except Exception as e:
            logger.warning("Local Ollama unavailable (%s) — using template fallback.", e)

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