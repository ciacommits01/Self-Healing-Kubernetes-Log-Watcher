# Self-Healing Kubernetes Log Watcher

An agent that tails EKS pod logs, detects anomalous crash-loop/error patterns
with an **unsupervised model trained from scratch**, takes automated
remediation action (restart pod / scale deployment / page a human), and uses
a **free, local LLM** (Ollama) to write a human-readable incident summary.

Built with synthetic seed data since no real cluster logs were available —
see [Why synthetic data, and does it generalize?](#why-synthetic-data-and-does-it-generalize) below.

```
watcher (tail logs)
    -> rolling per-pod feature windows
    -> IsolationForest anomaly scoring       (detector/)
    -> CRITICAL incident?
         - automated remediation action       (operator/k8s_actions.py)
         - human-readable summary              (llm/summarize.py)
         - page a human                        (operator/alerting.py)
    -> WARNING incident? -> logged only, no page, no action
```

## Quickstart (no cluster needed)

```bash
pip install -r requirements.txt

# 1. Generate synthetic "normal" logs + a "live" stream with 4 injected incidents
python data/generate_seed_logs.py

# 2. Train the anomaly detector on NORMAL logs only (never sees the incidents)
python detector/train.py --logs data/normal_logs.log --out detector/model.joblib

# 3. Run the full pipeline against the live stream
python main.py --mode simulate --log-file data/live_stream.log --model detector/model.joblib
```

You'll see the detector catch all 4 injected incidents (crash loop,
dependency timeout, OOMKill, error spike), take a dry-run remediation
action for each, and print a formatted page with an LLM (or fallback)
summary.

## How the detection model works

Rather than embedding raw log text (which needs a pretrained language
model), each log line is turned into a small set of hand-engineered,
SRE-recognizable signals inside a rolling **per-pod** window (default: last
8 lines):

| Feature | What it captures |
|---|---|
| `log_rate` | lines/sec — bursts show up here |
| `error_ratio` / `warn_ratio` | fraction of ERROR / WARN lines |
| `restart_signal` | "back-off", "terminated", "starting container" |
| `oom_signal` | "OOMKilled", "memory usage at N%" |
| `conn_signal` | "connection refused", "timeout", "503" |
| `panic_signal` | "panic", "exception", "nil pointer" |
| `distinct_error_types` | unique error messages / total errors |
| `avg_latency` / `latency_spike` | parsed request latency |

An **IsolationForest** (`scikit-learn`) is trained *only* on windows from
known-good logs — it never sees a single labeled anomaly. At inference,
every window gets an anomaly score; two thresholds (both derived from
percentiles of the training-score distribution) split findings into:

- **warning** — logged only, no page, no action (catches noisy-but-benign
  blips like one slow query)
- **critical** — triggers the automated action + LLM summary + page

This two-tier design exists because a single, looser threshold either pages
on every one-off slow request or misses real incidents — see the honest
numbers below.

## Evaluation on the synthetic live stream

4 incidents were injected (crash loop, dependency timeout, OOMKill, error
spike) into a 2,426-line stream. Result with the shipped model:

- **4/4 incidents detected at CRITICAL severity, each paged exactly once**
  (adjacent flagged windows for the same burst are merged so one incident
  doesn't generate duplicate pages)
- **1 false-positive CRITICAL page** (a slow-query spike that scored past
  the critical threshold) — precision isn't perfect, and shouldn't be
  assumed to be, on hand-crafted synthetic data
- Several more borderline WARNING-tier blips, correctly *not* paged

**Tunable knobs** (in `detector/train.py`): `window_size`, `stride`,
`contamination` (controls both thresholds). Retrain periodically on your
own "known-good" log window as real traffic patterns evolve.

## Why synthetic data, and does it generalize?

No EKS logs were available, so `data/generate_seed_logs.py` generates
realistic-shaped logs: normal request/health-check/GC traffic, plus 4
distinct incident templates (crash loop, OOMKill, dependency timeout, error
spike). The **feature extraction is format-aware, not content-memorized** —
it looks for generic signals (error ratio, restart/OOM/connection/panic
keywords, latency) rather than the exact synthetic wording, so it should
transfer reasonably well to real logs that use similar vocabulary.

That said, **your production log format will differ** (JSON logs, different
field order, different keywords for your stack). Before trusting this on a
real cluster:
1. Point `generate_seed_logs.py`'s templates at your real incident
   post-mortems and log formats
2. Replace/extend the `LOG_RE` regex and keyword lists in
   `detector/features.py` to match your actual log lines
3. Retrain on a real week of "known-good" logs from your cluster instead of
   the synthetic `normal_logs.log`
4. Run in `DRY_RUN=True` mode (the default in `operator/k8s_actions.py`)
   for a few days and compare pages against what you'd have wanted, before
   flipping to live remediation

## Connecting to a real EKS cluster

```bash
pip install kubernetes
aws eks update-kubeconfig --name <cluster-name> --region ap-south-1

python main.py --mode live --namespace prod --label-selector app=payment-api
```

`operator/watcher.py`'s `LiveK8sWatcher` streams real pod logs via the
official `kubernetes` Python client (`read_namespaced_pod_log(follow=True)`,
one thread per pod). It normalizes lines into the `<ts> <LEVEL> [<pod>]
<msg>` shape the feature extractor expects — **you'll likely need to adjust
this normalization** for your actual log format (JSON logs, Fluent Bit
output, etc).

Remediation actions in `operator/k8s_actions.py` are real Kubernetes API
calls (delete pod to trigger a replacement; patch deployment scale), guarded
by a **`DRY_RUN = True` default** — flip only after validating decisions.

## LLM summarization (free, local, no API key)

Uses [Ollama](https://ollama.com) — a free, open-weight local LLM runtime:

```bash
# install from https://ollama.com/download, then:
ollama pull llama3.2:1b
ollama serve   # usually auto-starts; serves localhost:11434
```

`llm/summarize.py` calls Ollama's local REST API. **If Ollama isn't
running** (as in this sandbox, or in CI), it falls back to a deterministic
template summary built from the same incident fields — so the pipeline
never silently drops a page just because the LLM is unavailable. Swap
`OLLAMA_MODEL` for any model you've pulled (`phi3`, `mistral`, etc).

## Project layout

```
data/
  generate_seed_logs.py   synthetic normal + incident log generator
  normal_logs.log         generated: training data (normal only)
  live_stream.log         generated: demo stream with 4 injected incidents
  anomaly_windows.json    ground truth for the injected incidents (eval only)
detector/
  features.py             log parsing + sliding-window feature extraction
  train.py                trains IsolationForest on normal_logs.log
  detect.py               scores a stream, groups + merges incidents
  model.joblib            generated: trained model + scaler + thresholds
operator/
  watcher.py              SimulatedWatcher (demo) + LiveK8sWatcher (real EKS)
  k8s_actions.py           restart_pod / scale_deployment (dry-run by default)
  alerting.py              Slack webhook or stdout fallback
llm/
  summarize.py             Ollama-based summary + deterministic fallback
main.py                    orchestrates the full pipeline
```

## Demo test apps (`crashy-app.yaml` vs `dependency-timeout-app.yaml`)

Two intentionally different failure modes, deployed to kind, to show what
this tool actually adds on top of Kubernetes' own self-healing.

### `crashy-app.yaml` — a real CrashLoopBackOff

A busybox container that logs ~20 normal `Handled GET /health` lines, then
one `panic:` line, then exits. Kubernetes detects the container has died
and restarts it on its own — that's standard kubelet behavior, nothing to
do with this project. This app exists to test two specific things:

- **`LiveK8sWatcher`'s reconnect logic** — a log stream closes every time
  a container exits, so the watcher has to detect that and reconnect with
  backoff, or it silently stops working the moment a real crash loop
  happens (the exact scenario it's meant to catch).
- **`PodStatusWatcher`** — most apps never print words like "OOMKilled" or
  "back-off restarting" themselves; those are facts Kubernetes knows about
  the container's lifecycle, not things the app logs. This component
  polls `restart_count` and `last_state.terminated.reason` directly from
  the K8s API and injects that ground truth into the same detection
  pipeline, since relying on log text alone under-detects real crash loops.

**Honest limitation:** since kubelet already restarts the container on its
own, our `restart_pod` remediation action here is mostly *redundant* with
what Kubernetes was already going to do. This app is useful for testing
the plumbing, not for demonstrating unique value.

### `dependency-timeout-app.yaml` — the actual "self-healing" demo

A busybox container that logs 5 normal lines, then logs `connection
refused` / `503` errors **forever, without ever exiting**. The container
stays `Running`, `RESTARTS` stays at `0` — kubelet has zero reason to do
anything, because nothing crashed. This is the scenario that shows this
project's real value: the anomaly detector reads the *log content* (not
container lifecycle events) and fires a `dependency_timeout` incident,
triggering `scale_deployment` — a real action Kubernetes itself would
never have taken, since from its point of view everything looks healthy.

**Verified result:** `kubectl get deployment checkout-worker` went from
`1/1` to `2/2` replicas, driven entirely by log-based detection, with the
container's health status never changing. That's the clearest before/after
proof of what this tool adds beyond stock Kubernetes.