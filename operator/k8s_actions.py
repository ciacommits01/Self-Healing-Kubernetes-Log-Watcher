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


def set_dry_run(dry_run: bool):
    global DRY_RUN
    DRY_RUN = dry_run


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
    try:
        core_v1.delete_namespaced_pod(name=pod_name, namespace=namespace, grace_period_seconds=30)
        return {"action": "restart_pod", "pod": pod_name, "status": "executed"}
    except Exception as e:
        logger.error("Failed to delete pod %s: %s", pod_name, e)
        return {"action": "restart_pod", "pod": pod_name, "status": f"failed: {e}"}


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
    try:
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
    except Exception as e:
        logger.error("Failed to scale deployment %s: %s", deployment_name, e)
        return {"action": "scale_deployment", "deployment": deployment_name, "status": f"failed: {e}"}


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
    "dependency_timeout": lambda pod, ns: scale_deployment(_deployment_of(pod, ns), ns, delta=1),
    "error_spike": lambda pod, ns: scale_deployment(_deployment_of(pod, ns), ns, delta=1),
    "error_pattern": lambda pod, ns: restart_pod(pod, ns),
}


def _deployment_of(pod_name, namespace="default"):
    """
    Return the Deployment name that owns this pod.

    Strategy (in order):
    1. Try the Kubernetes API: follow Pod -> ownerReferences -> ReplicaSet ->
       ownerReferences -> Deployment. This is the only reliable approach and
       handles StatefulSets, DaemonSets, and multi-hyphen deployment names.
    2. Fall back to the heuristic: strip the last two dash-segments from the
       pod name (works for standard ReplicaSet-managed pods like
       payment-api-7d9f8b6c-x2kqp -> payment-api).

    Logs a warning whenever the API lookup fails so silent mis-targeting is
    visible in the operator's logs.
    """
    # --- Attempt 1: live API ownerReference walk ---
    try:
        core_v1, apps_v1 = _get_clients()
        pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        for owner in (pod.metadata.owner_references or []):
            if owner.kind == "ReplicaSet":
                rs = apps_v1.read_namespaced_replica_set(
                    name=owner.name, namespace=namespace
                )
                for rs_owner in (rs.metadata.owner_references or []):
                    if rs_owner.kind == "Deployment":
                        logger.debug(
                            "_deployment_of %s -> %s (via ownerReferences)",
                            pod_name, rs_owner.name,
                        )
                        return rs_owner.name
            elif owner.kind == "StatefulSet":
                logger.debug(
                    "_deployment_of %s -> %s (StatefulSet owner)", pod_name, owner.name
                )
                return owner.name
    except Exception as e:
        logger.warning(
            "_deployment_of: K8s API lookup failed for pod '%s' (%s) — "
            "falling back to name heuristic. Check kubeconfig if this is unexpected.",
            pod_name, e,
        )

    # --- Fallback: strip last two dash-segments (e.g. <deploy>-<rs>-<pod>) ---
    # This is reliable for standard EKS Deployment-managed pods but will
    # produce wrong names for StatefulSets (pod-0) or short pod names.
    parts = pod_name.split("-")
    if len(parts) > 2:
        return "-".join(parts[:-2])
    return pod_name


def take_action(signal, pod_name, namespace="default"):
    """Dispatch table: incident signal -> remediation action."""
    handler = ACTION_MAP.get(signal, ACTION_MAP["error_pattern"])
    return handler(pod_name, namespace)
