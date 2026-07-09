"""
Real Kubernetes remediation actions, using the official `kubernetes` Python
client. These are the actions the watcher calls when a CRITICAL incident is
detected. Every action is a thin, auditable wrapper — nothing here is
"clever"; on purpose, so an SRE can read exactly what it will do to their
cluster before turning it on.

To actually run against an EKS cluster:
    pip install kubernetes
    aws eks update-kubeconfig --name <cluster-name> --region <region>
    (this populates ~/.kube/config, which load_kube_config() below reads)

DRY_RUN=True by default — actions are logged, not executed. Flip it only
once you've watched a few cycles of decisions in dry-run mode and trust them.
"""

import logging

logger = logging.getLogger("k8s_actions")

DRY_RUN = True  # flip to False only after validating decisions in dry-run mode


def _get_clients():
    from kubernetes import client, config

    try:
        config.load_incluster_config()  # running inside the cluster as an operator pod
    except Exception:
        config.load_kube_config()  # running from a laptop/CI against EKS via kubeconfig
    return client.CoreV1Api(), client.AppsV1Api()


def restart_pod(pod_name, namespace="default"):
    """
    Kubernetes has no native 'restart' verb for a single pod. The standard,
    safe pattern is to delete it: if it's managed by a Deployment/ReplicaSet/
    StatefulSet, the controller immediately schedules a fresh replacement.
    """
    logger.info("ACTION restart_pod pod=%s namespace=%s dry_run=%s", pod_name, namespace, DRY_RUN)
    if DRY_RUN:
        return {"action": "restart_pod", "pod": pod_name, "status": "dry_run"}

    core_v1, _ = _get_clients()
    core_v1.delete_namespaced_pod(name=pod_name, namespace=namespace, grace_period_seconds=30)
    return {"action": "restart_pod", "pod": pod_name, "status": "executed"}


def scale_deployment(deployment_name, namespace="default", delta=1, max_replicas=10):
    """
    Bumps replica count by `delta` (capped at max_replicas), e.g. to absorb
    load while a dependency recovers, or to spread a memory-pressure
    incident across more pods.
    """
    logger.info(
        "ACTION scale_deployment deployment=%s namespace=%s delta=%+d dry_run=%s",
        deployment_name, namespace, delta, DRY_RUN,
    )
    if DRY_RUN:
        return {"action": "scale_deployment", "deployment": deployment_name,
                 "delta": delta, "status": "dry_run"}

    _, apps_v1 = _get_clients()
    dep = apps_v1.read_namespaced_deployment(deployment_name, namespace)
    current = dep.spec.replicas or 1
    new_replicas = min(current + delta, max_replicas)
    apps_v1.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=namespace,
        body={"spec": {"replicas": new_replicas}},
    )
    return {
        "action": "scale_deployment",
        "deployment": deployment_name,
        "from": current,
        "to": new_replicas,
        "status": "executed",
    }


def cordon_node_if_repeated(pod_name, namespace="default"):
    """
    Placeholder for a higher-severity escalation: if the SAME pod keeps
    crash-looping across multiple restarts (tracked by the caller), cordon
    its node so the scheduler stops placing new pods there while a human
    investigates. Left as dry-run-only/logged by design — cordoning a node
    is disruptive and shouldn't be fully automatic without extra guardrails.
    """
    logger.warning("ESCALATION cordon_node_if_repeated pod=%s -- requires human confirmation", pod_name)
    return {"action": "cordon_node", "pod": pod_name, "status": "requires_human_confirmation"}


ACTION_MAP = {
    "crash_loop": lambda pod, ns: restart_pod(pod, ns),
    "oom_kill": lambda pod, ns: restart_pod(pod, ns),
    "dependency_timeout": lambda pod, ns: scale_deployment(_deployment_of(pod), ns, delta=1),
    "error_spike": lambda pod, ns: scale_deployment(_deployment_of(pod), ns, delta=1),
    "error_pattern": lambda pod, ns: restart_pod(pod, ns),
}


def _deployment_of(pod_name):
    """EKS pod names look like <deployment>-<replicaset-hash>-<pod-hash>;
    strip the last two dash-segments to recover the deployment name."""
    parts = pod_name.split("-")
    return "-".join(parts[:-2]) if len(parts) > 2 else pod_name


def take_action(signal, pod_name, namespace="default"):
    """Dispatch table: incident signal -> remediation action."""
    handler = ACTION_MAP.get(signal, ACTION_MAP["error_pattern"])
    return handler(pod_name, namespace)
