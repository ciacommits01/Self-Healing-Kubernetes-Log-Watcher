"""
Self-Healing Log Watcher - Monitoring Dashboard

Features:
  /                  Monitoring overview: stats cards, signal chart, pod health, incident feed
  /patterns          Drain3-mined log template browser
  /chat              Interactive SRE assistant
  /api/stats         JSON stats API
  /api/incidents     JSON incidents API (filterable)

Run:
    python dashboard/app.py
    # open http://localhost:5000
"""

import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "storage"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chatbot"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detector"))

import joblib  # noqa: E402
import markdown as md  # noqa: E402
from flask import Flask, render_template_string, request, jsonify  # noqa: E402

from incident_store import get_recent_incidents, get_stats  # noqa: E402
from chat_engine import ChatEngine  # noqa: E402
from template_miner import TemplateMinerWrapper  # noqa: E402

app = Flask(__name__)

_chat_engine = None


def get_chat_engine():
    global _chat_engine
    if _chat_engine is None:
        namespace = os.environ.get("DASHBOARD_NAMESPACE", "default")
        label_selector = os.environ.get("DASHBOARD_LABEL_SELECTOR")
        _chat_engine = ChatEngine(namespace=namespace, label_selector=label_selector)
    return _chat_engine


_UNICODE_NORMALIZE = {
    "\u2011": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}


def render_chat_text(text):
    for bad, good in _UNICODE_NORMALIZE.items():
        text = text.replace(bad, good)
    return md.markdown(text, extensions=["extra", "sane_lists"])


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "detector", "model.joblib")

_ERROR_KEYWORDS = re.compile(
    r"error|panic|exception|oom|killed|refused|timeout|deadline|back-off|"
    r"restart|terminated|traceback|unhandled|503|5\d\d",
    re.IGNORECASE,
)


def get_known_templates():
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


def _parse_action_label(action_json_str):
    try:
        obj = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
        act = obj.get("action", "action")
        if act == "restart_pod":
            return "restart pod"
        if act == "scale_deployment":
            delta = obj.get("delta", 1)
            return "scale +{}".format(delta)
        return act.replace("_", " ")
    except Exception:
        return "action"


app.jinja_env.globals["render_chat_text"] = render_chat_text
app.jinja_env.filters["parse_action"] = _parse_action_label
app.jinja_env.filters["tojson"] = json.dumps


BASE_STYLE = """
<style>
  :root {
    --bg:#0b0e14; --surface:#111520; --border:#1e2535; --border2:#2a3045;
    --text:#cdd6f4; --muted:#6c7a95; --accent:#89b4fa; --accent2:#74c7ec;
    --critical:#f38ba8; --warning:#f9e2af; --ok:#a6e3a1; --purple:#cba6f7;
    --r-crit:rgba(243,139,168,.08); --r-warn:rgba(249,226,175,.05);
  }
  *{box-sizing:border-box}
  body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    background:var(--bg);color:var(--text);margin:0;font-size:14px;line-height:1.5}
  .layout{display:flex;min-height:100vh}

  /* SIDEBAR */
  .sidebar{width:220px;min-height:100vh;flex-shrink:0;
    background:linear-gradient(180deg,#0d1117 0%,#111520 100%);
    border-right:1px solid var(--border);display:flex;flex-direction:column;
    padding:0 0 24px;position:sticky;top:0;height:100vh;overflow:auto}
  .sidebar-logo{padding:22px 20px 18px;border-bottom:1px solid var(--border);margin-bottom:12px}
  .logo-icon{font-size:22px;margin-bottom:6px}
  .logo-title{font-size:13px;font-weight:700;color:var(--text);line-height:1.3}
  .logo-sub{font-size:11px;color:var(--muted);margin-top:2px}
  .nav-section{padding:0 12px;margin-bottom:4px}
  .nav-label{font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:1px;padding:8px 8px 4px;display:block}
  .nav-link{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:7px;
    color:var(--muted);text-decoration:none;font-size:13px;font-weight:500;
    transition:background .15s,color .15s;margin-bottom:2px}
  .nav-link:hover{background:var(--border);color:var(--text)}
  .nav-link.active{background:rgba(137,180,250,.12);color:var(--accent)}
  .nav-icon{font-size:15px;width:18px;text-align:center}
  .sidebar-status{margin-top:auto;padding:12px 20px;border-top:1px solid var(--border);
    font-size:11px;color:var(--muted)}
  .status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:var(--ok);margin-right:5px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

  /* PAGE */
  .page{flex:1;padding:28px 32px;overflow:auto}
  .page-header{margin-bottom:24px}
  .page-header h2{margin:0 0 4px;font-size:20px;font-weight:700}
  .subtitle{color:var(--muted);font-size:12px}
  .refresh-hint{float:right;font-size:11px;color:var(--muted)}
  #countdown{color:var(--accent);font-weight:600}

  /* STAT CARDS */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:18px 20px;position:relative;overflow:hidden}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}
  .card.total::before{background:var(--accent)}
  .card.critical::before{background:var(--critical)}
  .card.warning::before{background:var(--warning)}
  .card.signals::before{background:var(--purple)}
  .card.pods::before{background:var(--ok)}
  .num{font-size:32px;font-weight:700;letter-spacing:-1px}
  .lbl{font-size:11px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
  .c-critical{color:var(--critical)}.c-warning{color:var(--warning)}.c-ok{color:var(--ok)}
  .c-accent{color:var(--accent)}.c-purple{color:var(--purple)}

  /* TWO-COL + PANELS */
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:28px}
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .panel-header{padding:14px 18px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between}
  .panel-title{font-size:13px;font-weight:600}
  .panel-body{padding:16px 18px}
  .tag-pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;
    background:var(--border);color:var(--muted)}

  /* SIGNAL BAR CHART */
  .signal-bars{display:flex;flex-direction:column;gap:10px}
  .signal-row{display:flex;align-items:center;gap:10px}
  .signal-label{width:130px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
  .signal-bar-wrap{flex:1;background:var(--border);border-radius:4px;height:8px;overflow:hidden}
  .signal-bar{height:100%;border-radius:4px;transition:width .4s}
  .signal-count{font-size:11px;color:var(--muted);width:24px;text-align:right;flex-shrink:0}
  .bar-crash_loop{background:var(--critical)}.bar-oom_kill{background:#fab387}
  .bar-dependency_timeout{background:var(--warning)}.bar-error_spike{background:#89dceb}
  .bar-error_pattern{background:var(--accent)}.bar-novel_pattern{background:var(--purple)}
  .bar-default{background:var(--muted)}

  /* POD HEALTH */
  .pod-health-table{width:100%;border-collapse:collapse}
  .pod-health-table th,.pod-health-table td{text-align:left;padding:8px 10px;font-size:12px;
    border-bottom:1px solid var(--border)}
  .pod-health-table th{color:var(--muted);font-weight:500;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
  .pod-health-table tr:last-child td{border:none}
  .pod-nm{font-family:ui-monospace,Consolas,monospace;font-size:11px}
  .heat-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
  .heat-critical{background:var(--critical)}.heat-warning{background:var(--warning)}.heat-ok{background:var(--ok)}
  .cov-bar{height:5px;background:var(--border);border-radius:3px;margin-top:5px;overflow:hidden}
  .cov-fill{height:100%;background:var(--accent);border-radius:3px}

  /* FILTER BAR */
  .filter-bar{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
  .filter-bar input[type=search],.filter-bar select{background:var(--surface);border:1px solid var(--border2);
    color:var(--text);padding:7px 12px;border-radius:7px;font-size:13px;outline:none;transition:border .15s}
  .filter-bar input[type=search]{flex:1;min-width:200px}
  .filter-bar input:focus,.filter-bar select:focus{border-color:var(--accent)}
  .filter-bar select option{background:var(--surface)}
  .filter-count{font-size:12px;color:var(--muted);margin-left:auto}

  /* INCIDENT TABLE */
  .itw{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .itable{width:100%;border-collapse:collapse}
  .itable th,.itable td{text-align:left;padding:10px 14px;font-size:12px;border-bottom:1px solid var(--border)}
  .itable th{color:var(--muted);font-weight:500;font-size:10px;text-transform:uppercase;
    letter-spacing:.5px;background:rgba(30,37,53,.5)}
  .itable tr:last-child td{border:none}
  .itable tbody tr{transition:background .12s}
  .itable tbody tr:hover{background:rgba(137,180,250,.04)}
  .row-critical{background:var(--r-crit)}.row-warning{background:var(--r-warn)}

  .badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;
    letter-spacing:.4px;text-transform:uppercase}
  .badge-critical{background:rgba(243,139,168,.15);color:var(--critical);border:1px solid rgba(243,139,168,.3)}
  .badge-warning{background:rgba(249,226,175,.10);color:var(--warning);border:1px solid rgba(249,226,175,.25)}
  .sig-chip{display:inline-block;padding:2px 8px;border-radius:5px;font-size:10px;font-weight:600;
    font-family:ui-monospace,Consolas,monospace;background:rgba(137,180,250,.10);color:var(--accent2);
    border:1px solid rgba(137,180,250,.15)}
  .score-cell{font-variant-numeric:tabular-nums;font-family:ui-monospace,Consolas,monospace;
    font-size:12px;color:var(--critical)}
  .pod-cell{font-family:ui-monospace,Consolas,monospace;max-width:160px;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .time-cell{color:var(--muted);font-size:11px;white-space:nowrap}
  .sum-toggle{cursor:pointer;color:var(--accent);font-size:11px}
  .sum-full{display:none;margin-top:6px;color:var(--muted);font-size:11px;line-height:1.5;max-width:340px}
  .sum-full.open{display:block}
  .act-chip{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;
    font-family:ui-monospace,Consolas,monospace;background:rgba(166,227,161,.08);color:var(--ok);
    border:1px solid rgba(166,227,161,.2);cursor:pointer}

  /* ACTION MODAL */
  .modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);
    z-index:100;align-items:center;justify-content:center}
  .modal-bg.open{display:flex}
  .modal{background:var(--surface);border:1px solid var(--border2);border-radius:12px;
    padding:24px;max-width:520px;width:90%}
  .modal h3{margin:0 0 14px;font-size:14px;color:var(--accent)}
  .modal pre{background:var(--bg);padding:12px;border-radius:8px;font-size:12px;
    overflow:auto;color:var(--ok);margin:0}
  .close-btn{margin-top:14px;background:var(--border);border:none;color:var(--text);
    padding:7px 16px;border-radius:6px;cursor:pointer;font-size:12px}
  .close-btn:hover{background:var(--border2)}

  /* CHAT */
  .chat-layout{display:flex;flex-direction:column;height:calc(100vh - 130px)}
  .chat-history{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;
    background:var(--surface);border:1px solid var(--border);border-radius:12px 12px 0 0}
  .chat-history::-webkit-scrollbar{width:5px}
  .chat-history::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
  .msg{max-width:80%}
  .msg.user{align-self:flex-end}.msg.assistant{align-self:flex-start}
  .bubble{padding:11px 15px;border-radius:14px;font-size:13px;line-height:1.6}
  .msg.user .bubble{background:rgba(137,180,250,.15);color:var(--accent);
    border:1px solid rgba(137,180,250,.2);border-radius:14px 14px 4px 14px}
  .msg.assistant .bubble{background:var(--bg);color:var(--text);
    border:1px solid var(--border);border-radius:14px 14px 14px 4px}
  .meta{font-size:10px;color:var(--muted);margin-top:4px;padding:0 4px}
  .msg.user .meta{text-align:right}
  .bubble p{margin:0 0 8px}.bubble p:last-child{margin:0}
  .bubble ul,.bubble ol{margin:0 0 8px;padding-left:20px}
  .bubble li{margin-bottom:3px}
  .bubble strong{color:#fff}
  .bubble code{background:rgba(255,255,255,.07);padding:1px 5px;border-radius:4px;font-size:11px}
  .chat-input-bar{background:var(--surface);border:1px solid var(--border);border-top:none;
    border-radius:0 0 12px 12px;padding:12px 14px;display:flex;gap:10px;align-items:center}
  .chat-input-bar input[type=text]{flex:1;background:var(--bg);border:1px solid var(--border2);
    color:var(--text);padding:10px 14px;border-radius:8px;font-size:13px;outline:none;transition:border .15s}
  .chat-input-bar input:focus{border-color:var(--accent)}
  .chat-input-bar input::placeholder{color:var(--muted)}
  .send-btn{background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;
    color:#0b0e14;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:700}
  .send-btn:hover{opacity:.88}
  .empty-chat{color:var(--muted);text-align:center;padding:40px;font-size:13px}
  .empty-chat .hint{font-size:12px;margin-top:8px;opacity:.6}

  /* PATTERNS */
  .pat-intro{background:rgba(137,180,250,.06);border:1px solid rgba(137,180,250,.15);
    border-radius:10px;padding:14px 18px;margin-bottom:20px;font-size:12px;color:var(--muted);line-height:1.6}
  .pat-intro strong{color:var(--text)}
  .pat-intro code{background:rgba(255,255,255,.07);padding:1px 5px;border-radius:4px}
  .ptw{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .ptable{width:100%;border-collapse:collapse}
  .ptable th,.ptable td{text-align:left;padding:10px 14px;font-size:12px;border-bottom:1px solid var(--border)}
  .ptable th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;background:rgba(30,37,53,.5)}
  .ptable tr:last-child td{border:none}
  .ptable tbody tr:hover{background:rgba(137,180,250,.04)}
  .tmpl{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--text);word-break:break-all}
  .ptag{display:inline-block;padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700}
  .ptag.normal{background:rgba(166,227,161,.10);color:var(--ok);border:1px solid rgba(166,227,161,.2)}
  .ptag.err{background:rgba(243,139,168,.10);color:var(--critical);border:1px solid rgba(243,139,168,.2)}

  .empty{color:var(--muted);padding:40px;text-align:center;font-size:13px}
</style>
"""

NAV_COMMON = """
<div class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-icon">&#129706;</div>
    <div class="logo-title">Self-Healing<br>Log Watcher</div>
    <div class="logo-sub">Kubernetes Operator</div>
  </div>
  <div class="nav-section">
    <span class="nav-label">Observe</span>
    <a href="/" class="nav-link {mon_active}"><span class="nav-icon">&#128202;</span> Monitoring</a>
    <a href="/patterns" class="nav-link {pat_active}"><span class="nav-icon">&#129528;</span> Log Patterns</a>
  </div>
  <div class="nav-section">
    <span class="nav-label">Investigate</span>
    <a href="/chat" class="nav-link {chat_active}"><span class="nav-icon">&#129302;</span> Chat Assistant</a>
  </div>
  <div class="sidebar-status">
    <span class="status-dot"></span> Pipeline active
  </div>
</div>
"""

DASHBOARD_TEMPLATE = """
<!doctype html><html lang="en">
<head><meta charset="utf-8"><title>Monitoring - Log Watcher</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{{ style|safe }}</head>
<body><div class="layout">
{{ nav|safe }}
<div class="page">
  <div class="page-header">
    <h2>&#128202; Monitoring Overview</h2>
    <div class="subtitle">
      All detected anomalies from the pipeline.
      <span class="refresh-hint">Auto-refresh in <span id="countdown">15</span>s</span>
    </div>
  </div>

  <div class="cards">
    <div class="card total"><div class="num c-accent">{{ stats.total }}</div><div class="lbl">Total Incidents</div></div>
    <div class="card critical"><div class="num c-critical">{{ stats.critical }}</div><div class="lbl">Critical</div></div>
    <div class="card warning"><div class="num c-warning">{{ stats.warning }}</div><div class="lbl">Warning</div></div>
    <div class="card signals"><div class="num c-purple">{{ stats.by_signal|length }}</div><div class="lbl">Signal Types</div></div>
    <div class="card pods"><div class="num c-ok">{{ stats.by_pod|length }}</div><div class="lbl">Affected Pods</div></div>
  </div>

  <div class="two-col">
    <div class="panel">
      <div class="panel-header"><span class="panel-title">&#128200; Signal Distribution</span><span class="tag-pill">{{ stats.total }} events</span></div>
      <div class="panel-body">
        {% if stats.by_signal %}
        <div class="signal-bars">
          {% set mc = stats.by_signal[0].c %}
          {% for s in stats.by_signal %}
          <div class="signal-row">
            <div class="signal-label" title="{{ s.signal }}">{{ s.signal }}</div>
            <div class="signal-bar-wrap"><div class="signal-bar bar-{{ s.signal }} bar-default" style="width:{{ (s.c/mc*100)|round|int }}%"></div></div>
            <div class="signal-count">{{ s.c }}</div>
          </div>
          {% endfor %}
        </div>
        {% else %}<div class="empty" style="padding:20px">No data yet.</div>{% endif %}
      </div>
    </div>

    <div class="panel">
      <div class="panel-header"><span class="panel-title">&#128308; Pod Health</span><span class="tag-pill">{{ stats.by_pod|length }} pods</span></div>
      <div class="panel-body" style="padding:0">
        {% if stats.by_pod %}
        <table class="pod-health-table">
          <thead><tr><th>Pod</th><th>Incidents</th><th>Share</th></tr></thead>
          <tbody>
          {% for p in stats.by_pod %}
          {% set pct = (p.c/stats.total*100)|round|int %}
          <tr>
            <td class="pod-nm" title="{{ p.pod }}">
              {% if p.c>=5 %}<span class="heat-dot heat-critical"></span>
              {% elif p.c>=2 %}<span class="heat-dot heat-warning"></span>
              {% else %}<span class="heat-dot heat-ok"></span>{% endif %}
              {{ p.pod[:26] }}{% if p.pod|length > 26 %}...{% endif %}
            </td>
            <td>{{ p.c }}</td>
            <td style="width:110px">
              <div class="cov-bar"><div class="cov-fill" style="width:{{ pct }}%;background:{% if p.c>=5 %}var(--critical){% elif p.c>=2 %}var(--warning){% else %}var(--ok){% endif %}"></div></div>
              <div style="font-size:10px;color:var(--muted);margin-top:2px">{{ pct }}%</div>
            </td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
        {% else %}<div class="empty" style="padding:20px">No pod data.</div>{% endif %}
      </div>
    </div>
  </div>

  <div style="margin-bottom:12px">
    <span style="font-size:15px;font-weight:700">&#128220; Incident Feed</span>
  </div>

  <div class="filter-bar">
    <input type="search" id="ft" placeholder="&#128269;  Filter by pod, signal, summary...">
    <select id="fs" onchange="af()">
      <option value="">All severities</option>
      <option value="critical">Critical only</option>
      <option value="warning">Warning only</option>
    </select>
    <select id="fg" onchange="af()">
      <option value="">All signals</option>
      {% for s in stats.by_signal %}<option value="{{ s.signal }}">{{ s.signal }}</option>{% endfor %}
    </select>
    <span class="filter-count" id="fc">{{ incidents|length }} incidents</span>
  </div>

  {% if incidents %}
  <div class="itw">
    <table class="itable" id="itable">
      <thead><tr><th>Time</th><th>Pod</th><th>Signal</th><th>Severity</th><th>Score</th><th>Action</th><th>Summary</th></tr></thead>
      <tbody>
      {% for inc in incidents %}
      <tr class="row-{{ inc.severity }}" data-pod="{{ inc.pod }}" data-signal="{{ inc.signal }}" data-sev="{{ inc.severity }}" data-summary="{{ inc.summary or '' }}">
        <td class="time-cell">{{ inc.detected_at[:19]|replace("T"," ") }}</td>
        <td class="pod-cell" title="{{ inc.pod }}">{{ inc.pod }}</td>
        <td><span class="sig-chip">{{ inc.signal }}</span></td>
        <td><span class="badge badge-{{ inc.severity }}">{{ inc.severity }}</span></td>
        <td class="score-cell">{{ "%.4f"|format(inc.score) }}</td>
        <td>
          {% if inc.action_json %}
          <span class="act-chip" onclick='showAct({{ inc.action_json|tojson }})'>{{ inc.action_json|parse_action }}</span>
          {% else %}<span style="color:var(--muted);font-size:11px">&#8212;</span>{% endif %}
        </td>
        <td>
          {% if inc.summary %}
          <span class="sum-toggle" onclick="ts(this)">View &#9660;</span>
          <div class="sum-full">{{ inc.summary }}</div>
          {% else %}<span style="color:var(--muted);font-size:11px">&#8212;</span>{% endif %}
        </td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div class="empty">No incidents yet. Run <code>main.py</code> to populate.</div>
  {% endif %}
</div>
</div>

<div class="modal-bg" id="amodal" onclick="if(event.target===this)cm()">
  <div class="modal">
    <h3>&#9881; Automated Action</h3>
    <pre id="ajson"></pre>
    <button class="close-btn" onclick="cm()">Close</button>
  </div>
</div>

<script>
  let t=15;const cd=document.getElementById('countdown');
  setInterval(()=>{t--;if(cd)cd.textContent=t;if(t<=0)location.reload()},1000);
  function af(){
    const text=document.getElementById('ft').value.toLowerCase();
    const sev=document.getElementById('fs').value;
    const sig=document.getElementById('fg').value;
    const rows=document.querySelectorAll('#itable tbody tr');
    let v=0;
    rows.forEach(r=>{
      const ok=(!text||(r.dataset.pod+r.dataset.signal+r.dataset.summary).toLowerCase().includes(text))
              &&(!sev||r.dataset.sev===sev)&&(!sig||r.dataset.signal===sig);
      r.style.display=ok?'':'none';if(ok)v++;
    });
    const fc=document.getElementById('fc');if(fc)fc.textContent=v+' incident'+(v!==1?'s':'');
  }
  document.getElementById('ft').addEventListener('input',af);
  function ts(el){const sf=el.nextElementSibling;sf.classList.toggle('open');el.textContent=sf.classList.contains('open')?'Hide \u25b2':'View \u25bc';}
  function showAct(raw){let o;try{o=typeof raw==='string'?JSON.parse(raw):raw;}catch(e){o=raw;}
    document.getElementById('ajson').textContent=JSON.stringify(o,null,2);
    document.getElementById('amodal').classList.add('open');}
  function cm(){document.getElementById('amodal').classList.remove('open');}
  document.addEventListener('keydown',e=>{if(e.key==='Escape')cm();});
</script>
</body></html>
"""

CHAT_TEMPLATE = """
<!doctype html><html lang="en">
<head><meta charset="utf-8"><title>Chat Assistant - Log Watcher</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{{ style|safe }}</head>
<body><div class="layout">
{{ nav|safe }}
<div class="page" style="display:flex;flex-direction:column;">
  <div class="page-header">
    <h2>&#129302; Cluster Health Assistant</h2>
    <div class="subtitle">Ask about incidents, pod status, anomaly patterns, or remediation actions.</div>
  </div>
  <form method="post" style="flex:1;display:flex;flex-direction:column;">
    <div class="chat-layout">
      <div class="chat-history" id="ch">
        {% if not history %}
        <div class="empty-chat">
          <div>&#128172; Start a conversation</div>
          <div class="hint">Try: "Which pod had the most incidents?" &bull; "Summarize the last 3 critical incidents"</div>
        </div>
        {% endif %}
        {% for role, text in history %}
        <div class="msg {{ 'user' if role == 'User' else 'assistant' }}">
          <div class="bubble">{% if role=='User' %}{{ text }}{% else %}{{ render_chat_text(text)|safe }}{% endif %}</div>
          <div class="meta">{{ role }}</div>
        </div>
        {% endfor %}
      </div>
      <div class="chat-input-bar">
        <input type="text" name="question" placeholder="e.g. What pods are most affected?" autocomplete="off" autofocus>
        <button type="submit" class="send-btn">Send &#8594;</button>
      </div>
    </div>
  </form>
</div>
</div>
<script>const ch=document.getElementById('ch');if(ch)ch.scrollTop=ch.scrollHeight;</script>
</body></html>
"""

PATTERNS_TEMPLATE = """
<!doctype html><html lang="en">
<head><meta charset="utf-8"><title>Log Patterns - Log Watcher</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{{ style|safe }}</head>
<body><div class="layout">
{{ nav|safe }}
<div class="page">
  <div class="page-header">
    <h2>&#129528; Log Template Patterns</h2>
    <div class="subtitle">Drain3-mined templates from training corpus.</div>
  </div>
  <div class="pat-intro">
    Message <strong>templates Drain3 mined</strong> from <code>data/normal_logs.log</code>.
    A shape <strong>not</strong> in this list is itself an anomaly signal.
    &nbsp;<strong>{{ total_templates }}</strong> templates &bull;
    <strong>{{ error_like_count }}</strong> error-like &bull;
    <strong>{{ normal_count }}</strong> normal
  </div>
  <div class="filter-bar" style="margin-bottom:16px">
    <input type="search" id="pf" placeholder="&#128269;  Filter templates...">
    <select id="pt" onchange="fp()">
      <option value="">All types</option>
      <option value="err">Error-like</option>
      <option value="normal">Normal</option>
    </select>
  </div>
  {% if not model_found %}
  <div class="empty">No trained model. Run <code>detector/train.py</code> first.</div>
  {% elif not templates %}
  <div class="empty">No templates mined. Check <code>detector/drain3_state.bin</code>.</div>
  {% else %}
  <div class="ptw">
    <table class="ptable" id="ptable">
      <thead><tr><th style="width:60px">ID</th><th>Template</th><th style="width:130px">Training lines</th><th style="width:100px">Type</th></tr></thead>
      <tbody>
      {% set ms = templates[0].size %}
      {% for t in templates %}
      <tr data-type="{{ 'err' if t.is_error_like else 'normal' }}" data-tmpl="{{ t.template|lower }}">
        <td style="color:var(--muted);font-family:monospace">{{ t.cluster_id }}</td>
        <td>
          <div class="tmpl">{{ t.template }}</div>
          <div class="cov-bar" style="margin-top:6px;max-width:260px">
            <div class="cov-fill" style="width:{{ (t.size/ms*100)|round|int }}%;background:{% if t.is_error_like %}var(--critical){% else %}var(--accent){% endif %}"></div>
          </div>
        </td>
        <td style="font-variant-numeric:tabular-nums">{{ t.size }}</td>
        <td>{% if t.is_error_like %}<span class="ptag err">error-like</span>{% else %}<span class="ptag normal">normal</span>{% endif %}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
</div>
</div>
<script>
  function fp(){
    const text=document.getElementById('pf').value.toLowerCase();
    const type=document.getElementById('pt').value;
    document.querySelectorAll('#ptable tbody tr').forEach(r=>{
      r.style.display=(!text||r.dataset.tmpl.includes(text))&&(!type||r.dataset.type===type)?'':'none';
    });
  }
  document.getElementById('pf').addEventListener('input',fp);
</script>
</body></html>
"""


def _make_nav(active):
    return NAV_COMMON.format(
        mon_active="active" if active == "mon" else "",
        pat_active="active" if active == "pat" else "",
        chat_active="active" if active == "chat" else "",
    )


@app.route("/")
def dashboard():
    stats = get_stats()
    incidents = get_recent_incidents(limit=100)
    return render_template_string(
        DASHBOARD_TEMPLATE, style=BASE_STYLE, nav=_make_nav("mon"),
        stats=stats, incidents=incidents,
    )


@app.route("/patterns")
def patterns():
    model_found, templates = get_known_templates()
    error_like_count = sum(1 for t in templates if t.get("is_error_like"))
    normal_count = len(templates) - error_like_count
    return render_template_string(
        PATTERNS_TEMPLATE, style=BASE_STYLE, nav=_make_nav("pat"),
        model_found=model_found, templates=templates,
        total_templates=len(templates),
        error_like_count=error_like_count,
        normal_count=normal_count,
    )


@app.route("/chat", methods=["GET", "POST"])
def chat():
    engine = get_chat_engine()
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            engine.ask(question)
    return render_template_string(
        CHAT_TEMPLATE, style=BASE_STYLE, nav=_make_nav("chat"),
        history=engine.history,
    )


@app.route("/api/stats")
def api_stats():
    """JSON endpoint for stat polling from external tools or JS."""
    return jsonify(get_stats())


@app.route("/api/incidents")
def api_incidents():
    """JSON incident feed. Optional ?severity=critical|warning&limit=N."""
    sev = request.args.get("severity")
    limit = int(request.args.get("limit", 50))
    return jsonify(get_recent_incidents(limit=limit, severity=sev))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
