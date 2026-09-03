# Self-Healing Kubernetes Log Watcher

An intelligent, log-driven Kubernetes operator that detects in-process application anomalies using an **unsupervised machine learning model trained from scratch**, executes automated self-healing remediation (scaling deployments, restarting pods), and generates human-readable incident root-cause summaries using a **local LLM (Ollama)** or fallback templates.

---

## The Problem: Why Kubernetes Alone Isn't Enough

A common misconception is that Kubernetes will automatically fix or scale applications when things go wrong. In reality:

1. **When a container crashes**:
   - Kubernetes (kubelet) only restarts the crashed container inside the *same* pod.
   - It **never** scales up replicas. If the failure persists, the pod enters `CrashLoopBackOff` while overall service capacity degrades.
2. **When a container suffers a "Silent Failure" (e.g., 503 errors, DB timeouts, deadlocks)**:
   - The application process is still alive and running inside its loop.
   - Kubelet reports the pod as **`Running` (Green ✅)** with `RESTARTS: 0`.
   - Metrics-based autoscalers (HPA) look at CPU/memory; if an error loop is I/O-bound or low-CPU, HPA does not scale.
   - To Kubernetes, everything looks healthy—even while **100% of user requests are failing**.

### How This Project Solves It

The **Self-Healing Kubernetes Log Watcher** monitors what Kubernetes cannot see: **semantic application log content**.

```
Live Pod Logs / Stream
       │
       ▼
Rolling Per-Pod Feature Windows (detector/features.py)
       │
       ▼
IsolationForest Anomaly Detector (detector/detect.py)
       │
   [ CRITICAL Anomaly Detected? ]
       ├─► Automated K8s Remediation (operator/k8s_actions.py)
       │    ├─ dependency_timeout / error_spike ──► Scale Deployment (+1 replica)
       │    └─ crash_loop / oom_kill ─────────────► Restart Pod (safe delete)
       │
       ├─► Root-Cause Summary via LLM (llm/summarize.py: Ollama / Fallback)
       │
       └─► Alert Notification (operator/alerting.py: Slack / stdout)
```

---

## Getting Started

### 1. Prerequisites & Virtual Environment

Clone the repository, create a virtual environment, and install dependencies:

```bash
git clone https://github.com/ciacommits01/Self-Healing-Kubernetes-Log-Watcher.git
cd Self-Healing-Kubernetes-Log-Watcher

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> Always ensure your virtual environment is activated (`source .venv/bin/activate`) before running commands, or invoke `./.venv/bin/python` directly.

---

## Quickstart 1: Simulation Mode (No Cluster Needed)

You can demo and evaluate the full detector, operator, and summarizer pipeline locally without a Kubernetes cluster.

```bash
# 1. Generate normal training logs + a live replay stream with 4 injected incidents
python data/generate_seed_logs.py

# 2. Train the anomaly detector on NORMAL logs only (unsupervised)
python detector/train.py --logs normal_logs.log --out detector/model.joblib

# 3. Run the full pipeline in simulation mode
python main.py --mode simulate --log-file live_stream.log --model detector/model.joblib
```

The pipeline will detect all 4 injected incidents (`crash_loop`, `dependency_timeout`, `oom_kill`, `error_spike`), output dry-run actions, and print formatted alerts with AI-generated incident summaries.

---

## Quickstart 2: Real-Time Live Testing on Kubernetes

You can run this against any local cluster (**Kind**, **Minikube**, **Docker Desktop**) or cloud provider (**EKS**, **GKE**, **AKS**).

### Setting Up a Local Cluster (Kind or Minikube)

If you already have Docker installed:

```bash
# Using Kind:
kind create cluster --name k8s-dev-cluster

# Or using Minikube:
minikube start --driver=docker
```

Verify your cluster connection:
```bash
kubectl cluster-info
kubectl get nodes
```

---

### Real-Time Test A: "Silent Failure" Self-Healing (`checkout-worker`)

This workload demonstrates active self-healing where stock Kubernetes fails to act. The container logs 5 normal requests, then logs `503` / `connection refused` errors in an infinite loop without crashing.

1. **Deploy the test application:**
   ```bash
   kubectl apply -f dependency-timeout-app.yaml
   ```
   Check that it starts with 1 replica:
   ```bash
   kubectl get deployment checkout-worker
   kubectl get pods -l app=checkout-worker
   ```

2. **Start the Log Watcher in Live Mode:**
   ```bash
   # Dry-run mode (safe inspection):
   python main.py --mode live --namespace default --label-selector app=checkout-worker --dry-run

   # Real remediation mode (actively executes Kubernetes API actions):
   python main.py --mode live --namespace default --label-selector app=checkout-worker --no-dry-run
   ```

3. **Observe the self-healing in real time:**
   - In your watcher terminal, you will see the anomaly detected:
     ```
     CRITICAL incident: pod=checkout-worker-... signal=dependency_timeout score=-0.6752
     ACTION scale_deployment deployment=checkout-worker namespace=default delta=+1 dry_run=False
     ```
   - In a separate terminal, watch the replicas scale up:
     ```bash
     kubectl get deployment checkout-worker -w
     ```
     You will observe the replica count increase from `1` to `2`, `3`, etc., driven completely by real-time log anomaly detection!

4. **Clean up:**
   ```bash
   kubectl delete -f dependency-timeout-app.yaml
   ```

---

### Real-Time Test B: CrashLoop Reconnection (`payment-api`)

This workload tests a container that logs 20 health checks, encounters a `panic:`, and exits.

1. **Deploy the crashing application:**
   ```bash
   kubectl apply -f crashy-app.yaml
   ```

2. **Run the watcher:**
   ```bash
   python main.py --mode live --namespace default --label-selector app=payment-api --dry-run
   ```

3. **Observe**:
   When the container terminates, the watcher catches the stream closure, applies exponential backoff, and seamlessly reconnects as soon as the container restarts.

4. **Clean up:**
   ```bash
   kubectl delete -f crashy-app.yaml
   ```

---

## How the Detection Model Works

Rather than embedding raw log text with heavy pretrained models, each log line is converted into SRE-recognizable numeric signals within a rolling per-pod window:

| Feature | What It Captures |
|---|---|
| `log_rate` | Log volume throughput (lines/second) — bursts show up here |
| `error_ratio` | Fraction of `ERROR` level lines in the window |
| `warn_ratio` | Fraction of `WARN` level lines in the window |
| `restart_signal` | Keyword matches for `"back-off"`, `"terminated"`, `"starting container"` |
| `oom_signal` | Keyword matches for `"OOMKilled"`, `"memory usage"` |
| `conn_signal` | Keyword matches for `"connection refused"`, `"timeout"`, `"503"` |
| `panic_signal` | Keyword matches for `"panic"`, `"exception"`, `"nil pointer"` |
| `distinct_error_types` | Ratio of unique error message templates to total errors |
| `avg_latency` | Average latency parsed from request log lines |
| `latency_spike` | Peak latency relative to window baseline |

An **IsolationForest** (`scikit-learn`) is trained *exclusively* on known-good logs (`normal_logs.log`). At runtime, each window receives an anomaly score and is categorized via two dynamic percentile thresholds:

- **`warning`**: Logged for observability; does not trigger disruptive actions or pages.
- **`critical`**: Triggers automated remediation + LLM incident summary + on-call page.

---

## LLM Incident Summarization

When a critical incident is detected, the pipeline automatically writes a concise incident summary for the on-call engineer:

1. **Local Ollama (Default, Offline & Free)**:
   ```bash
   # Optional: install Ollama from https://ollama.com
   ollama pull llama3.2:1b
   ollama serve
   ```
2. **Ollama Cloud (Optional)**:
   ```bash
   export OLLAMA_API_KEY=your_api_key
   ```
3. **Deterministic Fallback (Built-in)**:
   If Ollama is not installed or unreachable, the system automatically uses a built-in deterministic template. The pipeline **never drops an alert** due to an unavailable LLM.

---

## Project Layout

```
├── data/
│   └── generate_seed_logs.py      # Synthetic normal and incident log generator
├── detector/
│   ├── features.py                # Rolling per-pod window feature extraction
│   ├── train.py                   # IsolationForest training script
│   ├── detect.py                  # Anomaly scorer and incident clustering
│   └── model.joblib               # Trained model bundle (generated)
├── operator/
│   ├── watcher.py                 # SimulatedWatcher and dynamic LiveK8sWatcher
│   ├── k8s_actions.py             # Remediation actions (restart_pod, scale_deployment)
│   └── alerting.py                # Slack webhook and formatted stdout alerter
├── llm/
│   └── summarize.py               # Ollama local/cloud + template fallback summarizer
├── crashy-app.yaml                # Demo test app 1: Panic and crash-loop failure
├── dependency-timeout-app.yaml    # Demo test app 2: Silent 503 / connection failure
├── main.py                        # Pipeline orchestrator
└── requirements.txt               # Dependencies
```

---

## Configuration & Environment Variables

| Variable / Flag | Description | Default |
|---|---|---|
| `--mode` | Execution mode: `simulate` or `live` | `simulate` |
| `--dry-run` / `--no-dry-run` | Whether to execute remediation API calls or log them | `--dry-run` (True) |
| `--label-selector` | Kubernetes pod label selector for live mode | `app=payment-api` |
| `--namespace` | Kubernetes namespace to target | `default` |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL for Slack incident alerts | None (prints to stdout) |
| `OLLAMA_API_KEY` | Optional API key for Ollama Cloud | None (uses local Ollama / fallback) |