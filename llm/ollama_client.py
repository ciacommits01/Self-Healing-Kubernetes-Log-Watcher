"""
Shared Ollama calling logic — used by both llm/summarize.py (incident
summaries) and chatbot/chat_engine.py (interactive Q&A), so the
cloud/local/fallback chain lives in exactly one place.

Two modes, picked automatically based on OLLAMA_API_KEY:

  1. OLLAMA CLOUD: export OLLAMA_API_KEY=your_key (from
     https://ollama.com/settings/keys). Calls https://ollama.com/api/generate.
     No local install. Rate-limited/"free to start", not unconditionally free.

  2. LOCAL OLLAMA: ollama pull llama3.2:1b && ollama serve. Used whenever
     OLLAMA_API_KEY is NOT set. Fully offline, no per-request cost.

generate() returns (text, backend) where backend is 'ollama-cloud',
'ollama-local', or None if neither is reachable — callers decide their own
fallback behavior for the None case (summarize.py uses a template;
chat_engine.py tells the user honestly that no LLM is available, since a
canned template can't answer arbitrary questions).
"""

import logging
import os

import requests

logger = logging.getLogger("ollama_client")

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_CLOUD_URL = "https://ollama.com/api/generate"
OLLAMA_CLOUD_MODEL = "gpt-oss:120b"

OLLAMA_LOCAL_URL = "http://localhost:11434/api/generate"
OLLAMA_LOCAL_MODEL = "llama3.2:1b"


def _call_cloud(prompt, timeout=30):
    resp = requests.post(
        OLLAMA_CLOUD_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        json={"model": OLLAMA_CLOUD_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _call_local(prompt, timeout=15):
    resp = requests.post(
        OLLAMA_LOCAL_URL,
        json={"model": OLLAMA_LOCAL_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate(prompt):
    """
    Returns (text, backend). backend is 'ollama-cloud', 'ollama-local', or
    None (caller should apply their own fallback when None).
    """
    if OLLAMA_API_KEY:
        try:
            return _call_cloud(prompt), "ollama-cloud"
        except Exception as e:
            logger.warning("Ollama Cloud call failed (%s) — trying local Ollama next.", e)

    try:
        return _call_local(prompt), "ollama-local"
    except Exception as e:
        logger.warning("Local Ollama unavailable (%s).", e)

    return None, None
