"""
Seed data generator for the Self-Healing K8s Log Watcher.

Since we don't have a real EKS cluster to pull logs from, this generates
realistic pod log streams:
  - A long "normal" period used to TRAIN the anomaly detector (unsupervised,
    trained only on healthy behavior — it never sees labeled anomalies).
  - A "live" stream that mixes normal traffic with injected incident
    episodes (crash loops, OOMKills, dependency timeouts, error spikes)
    used to TEST/DEMO the detector + operator + LLM summary pipeline.

Ground-truth anomaly windows are saved alongside the live stream so you can
measure precision/recall, but the model itself never trains on them.
"""

import json
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

PODS = [
    "payment-api-7d9f8b6c-x2kqp",
    "payment-api-7d9f8b6c-w9m2z",
    "checkout-worker-5b7c9d-lk8fh",
    "inventory-svc-6f8b7c-p3n7v",
]

ROUTES = ["/health", "/checkout", "/payments/charge", "/inventory/lookup", "/cart"]

NORMAL_TEMPLATES = [
    ("INFO", "Handled GET {route} 200 {latency}ms"),
    ("INFO", "Handled POST {route} 201 {latency}ms"),
    ("INFO", "Handled GET {route} 200 {latency}ms"),
    ("DEBUG", "Cache hit for key session:{sid}"),
    ("INFO", "GC pause completed in {gc}ms, heap 41% used"),
    ("INFO", "Health check probe succeeded"),
    ("WARN", "Slow query detected: {latency}ms for SELECT on orders"),
]

# Each incident type is a short burst of characteristic log lines.
INCIDENTS = {
    "crash_loop": [
        ("ERROR", "panic: runtime error: invalid memory address or nil pointer dereference"),
        ("ERROR", "goroutine 42 [running]: main.handleRequest(...)"),
        ("INFO", "Container terminated, exit code 2"),
        ("WARN", "Back-off restarting failed container"),
        ("INFO", "Starting container payment-api"),
        ("ERROR", "panic: runtime error: invalid memory address or nil pointer dereference"),
        ("INFO", "Container terminated, exit code 2"),
        ("WARN", "Back-off restarting failed container"),
    ],
    "oom_kill": [
        ("WARN", "Memory usage at 92% of limit"),
        ("WARN", "Memory usage at 97% of limit"),
        ("ERROR", "Container payment-api killed: OOMKilled"),
        ("INFO", "Container terminated, exit code 137"),
        ("INFO", "Starting container payment-api"),
        ("WARN", "Memory usage at 89% of limit"),
    ],
    "dependency_timeout": [
        ("ERROR", "connection refused: dial tcp 10.0.4.12:5432: connect: connection refused"),
        ("ERROR", "connection refused: dial tcp 10.0.4.12:5432: connect: connection refused"),
        ("ERROR", "context deadline exceeded waiting for DB connection"),
        ("ERROR", "Handled POST /payments/charge 503 5002ms"),
        ("ERROR", "Handled POST /checkout 503 5011ms"),
        ("ERROR", "connection refused: dial tcp 10.0.4.12:5432: connect: connection refused"),
    ],
    "error_spike": [
        ("ERROR", "Handled POST /payments/charge 500 812ms"),
        ("ERROR", "Handled GET /inventory/lookup 500 733ms"),
        ("ERROR", "unhandled exception: KeyError('sku_id')"),
        ("ERROR", "Handled POST /checkout 500 690ms"),
        ("ERROR", "unhandled exception: KeyError('sku_id')"),
        ("ERROR", "Handled GET /cart 500 701ms"),
    ],
}


def fmt(ts, pod, level, msg):
    return f"{ts.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z {level:<5} [{pod}] {msg}"


def gen_normal_line(ts, pod):
    level, template = random.choice(NORMAL_TEMPLATES)
    msg = template.format(
        route=random.choice(ROUTES),
        latency=random.randint(8, 180),
        sid=random.randint(1000, 9999),
        gc=random.randint(5, 40),
    )
    return fmt(ts, pod, level, msg)


def gen_normal_block(start_ts, n_lines, step_seconds=0.4):
    lines = []
    ts = start_ts
    for _ in range(n_lines):
        pod = random.choice(PODS)
        lines.append(gen_normal_line(ts, pod))
        ts += timedelta(seconds=step_seconds * random.uniform(0.5, 1.5))
    return lines, ts


def gen_incident_block(start_ts, incident_type, pod, step_seconds=0.3):
    lines = []
    ts = start_ts
    for level, msg in INCIDENTS[incident_type]:
        lines.append(fmt(ts, pod, level, msg))
        ts += timedelta(seconds=step_seconds * random.uniform(0.6, 1.2))
    return lines, ts


def build_training_set(path="normal_logs.log", n_lines=6000):
    """Pure normal traffic only — this is what the detector trains on."""
    start = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    lines, _ = gen_normal_block(start, n_lines)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} normal training lines to {path}")


def build_live_stream(path="live_stream.log", labels_path="anomaly_windows.json"):
    """Normal traffic interleaved with 4 distinct incident episodes."""
    start = datetime(2026, 7, 6, 9, 0, 0, tzinfo=timezone.utc)
    ts = start
    all_lines = []
    ground_truth = []

    schedule = [
        ("normal", 400, None),
        ("incident", None, "crash_loop"),
        ("normal", 500, None),
        ("incident", None, "dependency_timeout"),
        ("normal", 600, None),
        ("incident", None, "oom_kill"),
        ("normal", 500, None),
        ("incident", None, "error_spike"),
        ("normal", 400, None),
    ]

    for kind, n_lines, incident_type in schedule:
        if kind == "normal":
            block, ts = gen_normal_block(ts, n_lines)
            all_lines.extend(block)
        else:
            pod = random.choice(PODS)
            incident_start_line = len(all_lines)
            block, ts = gen_incident_block(ts, incident_type, pod)
            all_lines.extend(block)
            ground_truth.append(
                {
                    "type": incident_type,
                    "pod": pod,
                    "start_line": incident_start_line,
                    "end_line": len(all_lines) - 1,
                    "start_ts": block[0].split(" ")[0],
                    "end_ts": block[-1].split(" ")[0],
                }
            )

    with open(path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    with open(labels_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {len(all_lines)} live-stream lines to {path}")
    print(f"Injected {len(ground_truth)} incidents -> {labels_path}")


if __name__ == "__main__":
    build_training_set()
    build_live_stream()
