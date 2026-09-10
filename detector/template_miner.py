"""
Drain3-based log template mining — a complementary feature source alongside
the hand-written keyword matching in features.py.

Why add this: keyword lists (RESTART_KEYWORDS, OOM_KEYWORDS, etc. in
features.py) only catch incident types someone thought to write a keyword
for. Drain3 (https://github.com/logpai/logparser's online variant) mines
log line TEMPLATES automatically — e.g. it learns "Handled GET /health 200
<*>ms" as one template from many slightly different lines, with zero
keyword list. During training it builds a vocabulary of "normal" templates
seen in known-good logs; during detection, a log line whose message
doesn't match any template from training is a genuine anomaly signal on
its own — even for incident types nobody wrote a keyword for.

Honest limitation, stated up front: on a log corpus with only a handful of
lexically distinct incident types (like this project's synthetic demo
data), the existing keyword features already separate them cleanly, so
this usually adds architectural value without moving the demo's detection
numbers. Its real payoff is on messy, high-cardinality PRODUCTION logs
with many services and formats you haven't hand-coded keywords for.

Persistence: Drain3's own FilePersistence writes cluster state to disk, so
the SAME state file used during training (detector/drain3_state.bin) is
reloaded during detection — meaning "novel template" during detection
means "never seen in the known-good training corpus," not "first time in
this process."
"""

import os

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


class TemplateMinerWrapper:
    def __init__(self, state_path, known_max_cluster_id=0):
        self.state_path = state_path
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.4  # similarity threshold for merging lines into one template
        config.drain_depth = 4
        persistence = FilePersistence(state_path)
        self.miner = TemplateMiner(persistence_handler=persistence, config=config)
        # If a state file already existed on disk, drain3 loaded it above —
        # known_max_cluster_id tells us where "training vocabulary" ends and
        # "genuinely new at detection time" begins.
        self.known_max_cluster_id = known_max_cluster_id

    def fit(self, messages):
        """
        Feed a corpus of NORMAL log messages (message text only — no
        timestamp/level/pod prefix) through the miner to build its
        vocabulary of known-good templates. Call once during training.
        Returns the max cluster_id reached, which the caller should persist
        (e.g. in the model bundle) and pass back in as known_max_cluster_id
        when reloading for detection.
        """
        for msg in messages:
            self.miner.add_log_message(msg)
        self.miner.save_state("post_training")
        self.known_max_cluster_id = max(
            (c.cluster_id for c in self.miner.drain.clusters), default=0
        )
        return self.known_max_cluster_id

    def process(self, msg):
        """
        Returns (cluster_id, is_novel) for one log message.

        is_novel=True means this message's cluster_id is higher than any
        cluster that existed right after training (a template that never
        appeared in the known-good corpus), OR drain3 just created a brand
        new cluster for it in this run — either way, a message shape the
        detector has never seen as "normal."
        """
        result = self.miner.add_log_message(msg)
        cluster_id = result["cluster_id"]
        is_novel = (
            cluster_id > self.known_max_cluster_id
            or result["change_type"] == "cluster_created"
        )
        return cluster_id, is_novel

    def known_templates(self):
        """Human-readable list of templates learned during training — used
        by the dashboard/chatbot to show what 'normal' looks like."""
        return [
            {"cluster_id": c.cluster_id, "template": c.get_template(), "size": c.size}
            for c in self.miner.drain.clusters
        ]
