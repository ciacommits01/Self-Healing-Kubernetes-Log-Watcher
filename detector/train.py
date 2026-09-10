"""
Trains the unsupervised anomaly detector on NORMAL logs only.

Model: IsolationForest over hand-engineered sliding-window features
(detector/features.py). This is trained from scratch on your own data —
no pretrained weights, no internet dependency. IsolationForest is a good
fit here because:
  - It's unsupervised (needs no anomaly labels).
  - It naturally handles the "mostly normal, rare weird points" shape of
    ops data.
  - It's cheap enough to retrain nightly on fresh "known-good" log windows
    as your system evolves.

We also fit a StandardScaler so features with different scales (e.g.
latency in ms vs. ratios in [0,1]) don't dominate the isolation splits.

Usage:
    python train.py --logs ../data/normal_logs.log --out model.joblib
"""

"""
Trains the unsupervised anomaly detector on NORMAL logs only.

Model: IsolationForest over two complementary feature sources:
  1. Hand-engineered keyword/ratio features (detector/features.py)
  2. Drain3 log template mining (detector/template_miner.py) — catches
     "a message shape never seen in normal logs" even for incident types
     nobody wrote a keyword for. See template_miner.py's docstring for the
     honest tradeoffs of this addition.

This is trained from scratch on your own data — no pretrained weights, no
internet dependency. IsolationForest is a good fit here because:
  - It's unsupervised (needs no anomaly labels).
  - It naturally handles the "mostly normal, rare weird points" shape of
    ops data.
  - It's cheap enough to retrain nightly on fresh "known-good" log windows
    as your system evolves.

We also fit a StandardScaler so features with different scales (e.g.
latency in ms vs. ratios in [0,1]) don't dominate the isolation splits.

Usage:
    python train.py --logs ../data/normal_logs.log --out model.joblib
"""

import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from features import FEATURE_NAMES, extract_windows, parse_line
from template_miner import TemplateMinerWrapper


def train(log_path, out_path, window_size=8, stride=2, contamination=0.015, drain_state_path=None):
    with open(log_path) as f:
        lines = f.readlines()

    if drain_state_path is None:
        drain_state_path = os.path.join(os.path.dirname(out_path) or ".", "drain3_state.bin")
    # Store as absolute so anything loading this bundle later (dashboard,
    # detect.py) resolves it correctly regardless of ITS OWN working
    # directory at runtime — a relative path here would only work if every
    # caller happened to run from the exact same cwd training used.
    drain_state_path = os.path.abspath(drain_state_path)
    # A stale state file from a previous training run would make "novel
    # template" comparisons meaningless (it'd compare against an old
    # vocabulary mixed with this run's). Start clean on every retrain.
    if os.path.exists(drain_state_path):
        os.remove(drain_state_path)

    template_miner = TemplateMinerWrapper(drain_state_path)
    messages = [p["msg"] for p in (parse_line(l) for l in lines) if p]
    known_max_cluster_id = template_miner.fit(messages)
    print(f"Drain3 mined {len(template_miner.known_templates())} templates from "
          f"{len(messages)} training messages (state: {drain_state_path})")

    windows = extract_windows(lines, window_size=window_size, stride=stride, template_miner=template_miner)
    if not windows:
        raise RuntimeError("No windows extracted — check log format / window_size.")

    X = np.array([w["features"] for w in windows])
    print(f"Extracted {len(X)} training windows with {X.shape[1]} features each.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination is set low: we're training on logs we BELIEVE are all
    # normal, so we only expect the model to flag its own natural outliers
    # (e.g. an unusually slow request), not real incidents.
    clf = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_features=1.0,
        random_state=42,
    )
    clf.fit(X_scaled)

    scores = clf.score_samples(X_scaled)
    # Two severity tiers, both derived from the SAME normal training
    # distribution:
    #   warn_threshold      (~1.5th percentile) -> "worth a look" / log it
    #   critical_threshold  (~0.3rd percentile) -> "page someone" / act
    # This lets the operator layer distinguish a genuinely rare pattern
    # (crash loops, OOMKills tend to score far into the tail) from
    # borderline noise (one slow request) without needing separate models.
    warn_threshold = np.percentile(scores, contamination * 100)
    critical_threshold = np.percentile(scores, contamination * 100 * 0.2)

    print(f"Training score stats -> mean: {scores.mean():.4f}, "
          f"min: {scores.min():.4f}, warn_threshold: {warn_threshold:.4f}, "
          f"critical_threshold: {critical_threshold:.4f}")

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "window_size": window_size,
        "stride": stride,
        "threshold": float(warn_threshold),
        "critical_threshold": float(critical_threshold),
        "drain_state_path": drain_state_path,
        "known_max_cluster_id": known_max_cluster_id,
    }
    joblib.dump(bundle, out_path)
    print(f"Saved trained detector to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="../data/normal_logs.log")
    p.add_argument("--out", default="model.joblib")
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--contamination", type=float, default=0.015)
    p.add_argument("--drain-state", default=None, help="Path for Drain3 persisted state (default: alongside --out)")
    args = p.parse_args()
    train(args.logs, args.out, args.window_size, args.stride, args.contamination, args.drain_state)
