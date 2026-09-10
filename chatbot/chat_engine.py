"""
Chatbot engine for asking questions about cluster health, recent
incidents, and log patterns — reusing the same Ollama key/chain as the
incident summarizer (llm/ollama_client.py).

Deliberately scoped: this is ONE context-stuffed LLM call per turn, not an
autonomous agent — it gathers real data (recent incidents from SQLite,
live pod status from the K8s API) and hands it to the model to answer
from, with a short bounded conversation history for continuity. It does
NOT do multi-step tool use, and it explicitly does NOT attempt genuine
forecasting/prediction — the prompt tells the model to say so honestly
rather than speculate, since real incident prediction (as opposed to
"what does the history show") is a distinct, much bigger project.
"""

import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "storage"))

from ollama_client import generate  # noqa: E402
from incident_store import get_recent_incidents, get_stats  # noqa: E402

logger = logging.getLogger("chat_engine")

MAX_HISTORY_TURNS = 5  # bounded so the prompt doesn't grow unbounded over a long session

SYSTEM_PROMPT = """You are an SRE assistant embedded in a Kubernetes self-healing log watcher tool. \
You answer questions about this specific cluster's health, recent incidents, and log patterns, \
using ONLY the data provided below.

Rules:
- If the data below doesn't cover what's being asked, say so honestly rather than guessing.
- Do NOT predict or forecast future incidents. You may describe patterns in the historical data \
if asked (e.g. "which pod has had the most incidents"), but do not claim to know what will happen next.
- Be direct and concise. No filler, no markdown headers, conversational tone.

CURRENT POD STATUS:
{pod_status}

INCIDENT HISTORY STATS:
{stats_block}

RECENT INCIDENTS (most recent first, up to 10):
{incidents_block}
"""


def get_pod_status_text(namespace="default", label_selector=None):
    """
    Live, READ-ONLY pod status from the K8s API — no actions taken here.
    Returns a plain-text summary, or an honest "unavailable" message if no
    cluster is reachable (e.g. kind isn't running, or no kubeconfig) — the
    chatbot should still work for pure incident-history questions even
    without a live cluster.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        core_v1 = client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(namespace, label_selector=label_selector)
        if not pods.items:
            return f"No pods found in namespace '{namespace}'" + (f" matching '{label_selector}'" if label_selector else "") + "."

        lines = []
        for pod in pods.items:
            phase = pod.status.phase
            restarts = sum(cs.restart_count for cs in (pod.status.container_statuses or []))
            ready = all(cs.ready for cs in (pod.status.container_statuses or [])) if pod.status.container_statuses else False
            lines.append(f"- {pod.metadata.name}: phase={phase}, ready={ready}, restarts={restarts}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Could not fetch live pod status: %s", e)
        return f"(Live pod status unavailable: {e}. Answering from incident history only.)"


def _format_stats(stats):
    lines = [
        f"Total incidents logged: {stats['total']} ({stats['critical']} critical, {stats['warning']} warning)",
    ]
    if stats["by_signal"]:
        signal_str = ", ".join(f"{r['signal']}={r['c']}" for r in stats["by_signal"])
        lines.append(f"By signal type: {signal_str}")
    if stats["by_pod"]:
        pod_str = ", ".join(f"{r['pod']}={r['c']}" for r in stats["by_pod"][:5])
        lines.append(f"Most-affected pods: {pod_str}")
    return "\n".join(lines)


def _format_incidents(incidents):
    if not incidents:
        return "(none recorded yet)"
    lines = []
    for inc in incidents[:10]:
        lines.append(
            f"- [{inc['detected_at']}] {inc['severity'].upper()} on {inc['pod']}: "
            f"signal={inc['signal']}, score={inc['score']:.3f}"
            + (f" — {inc['summary'][:150]}" if inc.get("summary") else "")
        )
    return "\n".join(lines)


class ChatEngine:
    def __init__(self, namespace="default", label_selector=None):
        self.namespace = namespace
        self.label_selector = label_selector
        self.history = []  # list of (role, text) tuples, bounded

    def ask(self, question):
        pod_status = get_pod_status_text(self.namespace, self.label_selector)
        stats = get_stats()
        incidents = get_recent_incidents(limit=10)

        system_context = SYSTEM_PROMPT.format(
            pod_status=pod_status,
            stats_block=_format_stats(stats),
            incidents_block=_format_incidents(incidents),
        )

        history_block = ""
        for role, text in self.history[-MAX_HISTORY_TURNS:]:
            history_block += f"{role}: {text}\n"

        prompt = f"{system_context}\nCONVERSATION SO FAR:\n{history_block}\nUser: {question}\nAssistant:"

        text, backend = generate(prompt)
        if text is None:
            text = (
                "No LLM is reachable right now (no OLLAMA_API_KEY set and no local "
                "Ollama running at localhost:11434). Set one of those and try again — "
                "I can't answer this without a language model backing me."
            )
            backend = "none"

        self.history.append(("User", question))
        self.history.append(("Assistant", text))
        return text, backend


if __name__ == "__main__":
    # Interactive CLI — see chatbot/chat_cli.py for the polished entry point.
    # This is just a quick manual smoke test.
    engine = ChatEngine()
    answer, backend = engine.ask("How many incidents have been recorded, and what's the most common signal?")
    print(f"[backend={backend}]\n{answer}")
