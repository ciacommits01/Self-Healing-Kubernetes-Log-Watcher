"""
Tails pod logs and feeds them, line by line, into the detector's rolling
windows in near-real-time. Two modes:

  - LiveK8sWatcher: streams real logs from EKS via the kubernetes Python
    client (works with any Deployment/label selector).
  - SimulatedWatcher: replays a log file at a configurable speed, so the
    full detector -> operator -> LLM pipeline can be demoed and tested
    without a real cluster.

Both feed the same `on_line(line)` callback, so main.py doesn't care which
mode it's running in.
"""

import logging
import re
import time

logger = logging.getLogger("watcher")

# Mirrors detector/features.py's LOG_RE, kept local (not imported) so this
# module works standalone even if sys.path isn't wired up by main.py.
_ALREADY_SHAPED_RE = re.compile(r"^\S+\s+\w+\s+\[[^\]]+\]\s+.*$")


class SimulatedWatcher:
    def __init__(self, log_path, speed=0.0):
        """speed=0.0 replays as fast as possible (good for batch demo runs);
        set e.g. speed=0.01 to simulate near-real-time streaming."""
        self.log_path = log_path
        self.speed = speed

    def run(self, on_line):
        with open(self.log_path) as f:
            for line in f:
                on_line(line)
                if self.speed:
                    time.sleep(self.speed)


class LiveK8sWatcher:
    """
    Streams logs from every pod matching a label selector in a namespace,
    using the kubernetes client's follow=True log stream (one thread per
    pod). Requires: pip install kubernetes, and a valid kubeconfig
    (aws eks update-kubeconfig ...) or in-cluster service account.
    """

    def __init__(self, namespace, label_selector):
        self.namespace = namespace
        self.label_selector = label_selector

    def _stream_pod_logs(self, core_v1, pod_name, on_line, stop_event):
        """
        Streams one pod's logs, reconnecting whenever the stream closes.

        A log stream closes any time the CONTAINER it's reading from exits
        — which includes the exact crash-loop case this whole project is
        built to detect. When that happens, Kubernetes starts a brand new
        container instance whose log stream begins from scratch anyway, so
        no de-dup is needed there. But a reconnect can also happen for a
        boring reason (network hiccup on a container that's still running),
        and in THAT case reconnecting with no filter would replay the whole
        log from the container's start again. `since_seconds` (the actual
        parameter the kubernetes client exposes — there is no `since_time`
        despite the raw K8s API supporting `sinceTime`) handles that case by
        only asking for logs newer than our last successful read.
        """
        import time

        last_read_epoch = None
        backoff = 1

        while not stop_event.is_set():
            try:
                kwargs = dict(
                    name=pod_name,
                    namespace=self.namespace,
                    follow=True,
                    _preload_content=False,
                    timestamps=False,  # your container's own log line already
                                        # carries a timestamp in most real
                                        # setups; k8s's own prefix would
                                        # otherwise double up and break parsing
                )
                if last_read_epoch is not None:
                    elapsed = int(time.time() - last_read_epoch) + 1  # +1s safety margin
                    kwargs["since_seconds"] = max(elapsed, 1)

                stream = core_v1.read_namespaced_pod_log(**kwargs)
                for raw_line in stream:
                    line = self._normalize_line(raw_line, pod_name)
                    if line is None:
                        continue
                    last_read_epoch = time.time()
                    on_line(line)
                    backoff = 1  # reset once we've seen real data again

                logger.warning(
                    "Log stream for pod %s closed (likely a container "
                    "restart) — reconnecting in %ss", pod_name, backoff,
                )
            except Exception as e:
                logger.error("Log stream error for pod %s: %s — retrying in %ss", pod_name, e, backoff)

            if stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)  # exponential backoff, capped at 30s

    def _normalize_line(self, raw_line, pod_name):
        """
        Only wraps lines that DON'T already match the detector's expected
        "<ts> <LEVEL> [<pod>] <msg>" shape. If your container already logs
        in (or close to) that shape — as our kind demo app does — this is
        a pure passthrough. If it logs plain unstructured text, we wrap it
        with a reception-time timestamp, a default INFO level, and the pod
        tag, so features.py can still parse it.

        For real production logs (JSON, structured, etc.) replace this
        method with a small adapter for your actual format.
        """
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            return None
        if _ALREADY_SHAPED_RE.match(line):
            return line + "\n"
        return f"{_now_rfc3339()} INFO  [{pod_name}] {line}\n"

    def run(self, on_line):
        from kubernetes import client, config
        import threading

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        core_v1 = client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(
            self.namespace, label_selector=self.label_selector
        )
        stop_event = threading.Event()
        threads = []
        for pod in pods.items:
            t = threading.Thread(
                target=self._stream_pod_logs,
                args=(core_v1, pod.metadata.name, on_line, stop_event),
                daemon=True,
            )
            t.start()
            threads.append(t)

        logger.info("Streaming logs from %d pod(s) — reconnects automatically on restart. Ctrl+C to stop.", len(threads))
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("Stopping watcher...")
            stop_event.set()


def _now_rfc3339():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
