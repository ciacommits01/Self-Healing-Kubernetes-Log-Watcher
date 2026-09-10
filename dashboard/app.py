"""
Simple read-only monitoring dashboard + chat page over the incident
history SQLite store.

Deliberately scoped down, per explicit agreement before building this:
  - No auth, no multi-user, no real-time push (websockets) — a single
    local Flask process with a plain HTML meta-refresh on the monitoring
    page is enough for a personal/demo tool.
  - No destructive actions in the UI (no "clear history" button) — that
    lives in storage/incident_store.py's clear_all(), callable directly
    if you want a clean slate between demo runs.
  - Chat history is a single in-memory ChatEngine shared across requests
    (fine for a single-user local tool; would need per-session state for
    anything multi-user).

Run:
    python dashboard/app.py
    # then open http://localhost:5000
"""

import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "storage"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chatbot"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detector"))

import joblib  # noqa: E402
import markdown as md  # noqa: E402
from flask import Flask, render_template_string, request  # noqa: E402

from incident_store import get_recent_incidents, get_stats  # noqa: E402
from chat_engine import ChatEngine  # noqa: E402
from template_miner import TemplateMinerWrapper  # noqa: E402

app = Flask(__name__)

# Single shared engine — see module docstring on why this is fine for a
# local single-user tool and not for anything multi-user.
_chat_engine = None


def get_chat_engine():
    global _chat_engine
    if _chat_engine is None:
        namespace = os.environ.get("DASHBOARD_NAMESPACE", "default")
        label_selector = os.environ.get("DASHBOARD_LABEL_SELECTOR")  # e.g. "app=checkout-worker"
        _chat_engine = ChatEngine(namespace=namespace, label_selector=label_selector)
    return _chat_engine


# Models (gpt-oss especially) tend to write "-" as a typographic dash
# (‑ U+2011, – U+2013, — U+2014) and curly quotes, which some terminals/
# fonts render as a broken box or look like a typo rather than a dash.
# Normalize to plain ASCII before rendering, on top of actual markdown
# rendering for **bold**/bullets so they show as real HTML, not literal
# asterisks.
_UNICODE_NORMALIZE = {
    "\u2011": "-", "\u2013": "-", "\u2014": "-",  # non-breaking/en/em dash
    "\u2018": "'", "\u2019": "'",                  # curly single quotes
    "\u201c": '"', "\u201d": '"',                  # curly double quotes
}


def render_chat_text(text):
    for bad, good in _UNICODE_NORMALIZE.items():
        text = text.replace(bad, good)
    return md.markdown(text, extensions=["extra", "sane_lists"])


BASE_STYLE = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #0f1115; color: #e4e6eb; margin: 0; }
  header { background: #171a21; padding: 16px 24px; border-bottom: 1px solid #2a2e37; display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 18px; margin: 0; }
  nav a { color: #9aa4b2; text-decoration: none; margin-left: 20px; font-size: 14px; }
  nav a.active, nav a:hover { color: #e4e6eb; }
  main { padding: 24px; max-width: 1000px; margin: 0 auto; }
  .cards { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .card { background: #171a21; border: 1px solid #2a2e37; border-radius: 8px; padding: 16px 20px; min-width: 140px; }
  .card .num { font-size: 28px; font-weight: 600; }
  .card .label { font-size: 12px; color: #9aa4b2; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; background: #171a21; border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 10px 14px; font-size: 13px; border-bottom: 1px solid #2a2e37; }
  th { color: #9aa4b2; font-weight: 500; text-transform: uppercase; font-size: 11px; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-critical { background: #3a1a1a; color: #ff6b6b; }
  .badge-warning { background: #3a331a; color: #ffd166; }
  .summary-cell { max-width: 320px; color: #9aa4b2; font-size: 12px; }
  .chat-box { background: #171a21; border: 1px solid #2a2e37; border-radius: 8px; padding: 20px; }
  .chat-history { max-height: 500px; overflow-y: auto; margin-bottom: 16px; }
  .msg { margin-bottom: 14px; }
  .msg .role { font-size: 11px; color: #9aa4b2; text-transform: uppercase; margin-bottom: 4px; }
  .msg .text { line-height: 1.6; }
  .msg .text p { margin: 0 0 10px 0; }
  .msg .text ul, .msg .text ol { margin: 0 0 10px 0; padding-left: 22px; }
  .msg .text li { margin-bottom: 4px; }
  .msg .text strong { color: #fff; }
  .msg .text code { background: #0f1115; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  .msg.user .text { color: #7dd3fc; }
  .pattern-tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #1a2e3a; color: #7dd3fc; }
  .pattern-tag.error-like { background: #3a1a1a; color: #ff6b6b; }
  .template-text { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
  form { display: flex; gap: 8px; }
  input[type=text] { flex: 1; background: #0f1115; border: 1px solid #2a2e37; color: #e4e6eb; padding: 10px 12px; border-radius: 6px; font-size: 14px; }
  button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:hover { background: #2563eb; }
  .empty { color: #9aa4b2; padding: 30px; text-align: center; }
</style>
"""

DASHBOARD_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Log Watcher Dashboard</title>
  <meta http-equiv="refresh" content="15">
  {{ style|safe }}
</head>
<body>
  <header>
    <h1>🩺 Self-Healing Log Watcher</h1>
    <nav>
      <a href="/" class="active">Monitoring</a>
      <a href="/patterns">Error Patterns</a>
      <a href="/chat">Chat</a>
    </nav>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="num">{{ stats.total }}</div><div class="label">Total incidents</div></div>
      <div class="card"><div class="num" style="color:#ff6b6b">{{ stats.critical }}</div><div class="label">Critical</div></div>
      <div class="card"><div class="num" style="color:#ffd166">{{ stats.warning }}</div><div class="label">Warning</div></div>
      <div class="card"><div class="num">{{ stats.by_signal|length }}</div><div class="label">Distinct signal types</div></div>
    </div>

    {% if incidents %}
    <table>
      <tr><th>Detected</th><th>Pod</th><th>Signal</th><th>Severity</th><th>Score</th><th>Action</th><th>Summary</th></tr>
      {% for inc in incidents %}
      <tr>
        <td>{{ inc.detected_at[:19] }}</td>
        <td>{{ inc.pod }}</td>
        <td>{{ inc.signal }}</td>
        <td><span class="badge badge-{{ inc.severity }}">{{ inc.severity|upper }}</span></td>
        <td>{{ "%.3f"|format(inc.score) }}</td>
        <td>{{ (inc.action_json or '—')[:60] }}</td>
        <td class="summary-cell">{{ (inc.summary or '—')[:150] }}</td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty">No incidents recorded yet. Run <code>main.py</code> against a log stream to populate this.</div>
    {% endif %}
  </main>
</body>
</html>
"""

CHAT_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Cluster Health Chat</title>
  {{ style|safe }}
</head>
<body>
  <header>
    <h1>🩺 Self-Healing Log Watcher</h1>
    <nav>
      <a href="/">Monitoring</a>
      <a href="/patterns">Error Patterns</a>
      <a href="/chat" class="active">Chat</a>
    </nav>
  </header>
  <main>
    <div class="chat-box">
      <div class="chat-history">
        {% if not history %}
        <div class="empty">Ask about recent incidents, current pod status, or log patterns.</div>
        {% endif %}
        {% for role, text in history %}
        <div class="msg {{ 'user' if role == 'User' else 'assistant' }}">
          <div class="role">{{ role }}</div>
          {% if role == 'User' %}
          <div class="text">{{ text }}</div>
          {% else %}
          <div class="text">{{ render_chat_text(text)|safe }}</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
      <form method="post">
        <input type="text" name="question" placeholder="e.g. What's the current status of my pods?" autofocus>
        <button type="submit">Send</button>
      </form>
    </div>
  </main>
</body>
</html>
"""

PATTERNS_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Error Patterns — Log Watcher</title>
  {{ style|safe }}
</head>
<body>
  <header>
    <h1>🩺 Self-Healing Log Watcher</h1>
    <nav>
      <a href="/">Monitoring</a>
      <a href="/patterns" class="active">Error Patterns</a>
      <a href="/chat">Chat</a>
    </nav>
  </header>
  <main>
    <p style="color:#9aa4b2; font-size:13px; margin-bottom:20px; line-height:1.6;">
      These are the log message <strong style="color:#e4e6eb;">templates Drain3 mined</strong>
      from the training corpus (<code>data/normal_logs.log</code>) — the vocabulary the detector's
      <code>novel_template_ratio</code> feature compares live log lines against. A message shape
      NOT in this list during detection is itself an anomaly signal, independent of keyword matching.
      Templates whose text matches known incident keywords (error/panic/oom/timeout/etc.) are flagged below —
      note most templates here come from NORMAL training data, so this list is mostly what
      "healthy" looks like, not a list of active errors.
    </p>
    {% if not model_found %}
    <div class="empty">No trained model found at detector/model.joblib yet. Run detector/train.py first.</div>
    {% elif not templates %}
    <div class="empty">Model found, but no templates were mined (unexpected — check detector/drain3_state.bin).</div>
    {% else %}
    <table>
      <tr><th>Cluster ID</th><th>Template</th><th>Lines matched (training)</th><th>Type</th></tr>
      {% for t in templates %}
      <tr>
        <td>{{ t.cluster_id }}</td>
        <td class="template-text">{{ t.template }}</td>
        <td>{{ t.size }}</td>
        <td>
          {% if t.is_error_like %}
          <span class="pattern-tag error-like">error-like</span>
          {% else %}
          <span class="pattern-tag">normal</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}
  </main>
</body>
</html>
"""


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "detector", "model.joblib")

_ERROR_KEYWORDS = re.compile(
    r"error|panic|exception|oom|killed|refused|timeout|deadline|back-off|"
    r"restart|terminated|traceback|unhandled|503|5\d\d",
    re.IGNORECASE,
)


def get_known_templates():
    """Loads the trained model's Drain3 vocabulary, read-only — does not
    mutate detector/drain3_state.bin. Returns (model_found, templates)."""
    if not os.path.exists(MODEL_PATH):
        return False, []
    bundle = joblib.load(MODEL_PATH)
    drain_state_path = bundle.get("drain_state_path")
    known_max_cluster_id = bundle.get("known_max_cluster_id", 0)
    if not drain_state_path or not os.path.exists(drain_state_path):
        return True, []

    miner = TemplateMinerWrapper(drain_state_path, known_max_cluster_id)
    templates = miner.known_templates()
    for t in templates:
        t["is_error_like"] = bool(_ERROR_KEYWORDS.search(t["template"]))
    templates.sort(key=lambda t: (-t["is_error_like"], -t["size"]))
    return True, templates


app.jinja_env.globals["render_chat_text"] = render_chat_text


@app.route("/")
def dashboard():
    stats = get_stats()
    incidents = get_recent_incidents(limit=50)
    return render_template_string(DASHBOARD_TEMPLATE, style=BASE_STYLE, stats=stats, incidents=incidents)


@app.route("/patterns")
def patterns():
    model_found, templates = get_known_templates()
    return render_template_string(PATTERNS_TEMPLATE, style=BASE_STYLE, model_found=model_found, templates=templates)


@app.route("/chat", methods=["GET", "POST"])
def chat():
    engine = get_chat_engine()
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            engine.ask(question)
    return render_template_string(CHAT_TEMPLATE, style=BASE_STYLE, history=engine.history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
