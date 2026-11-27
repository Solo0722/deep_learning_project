"""Class-pooled retention gates. No detector, loss maximization or training.

Input is the immutable per-object v1 probe schema. This is a reusable decision
component; applying it to archived observations is only a retrospective replay.
"""
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math

import numpy as np

OPERATIONS = ("brightness_down", "brightness_up", "contrast_down", "contrast_up",
              "gamma_brighten", "gamma_darken", "gaussian_blur", "gaussian_noise")
ACTIONS = tuple((op, level) for op in OPERATIONS for level in (1, 2, 3))
SETS = tuple(s for n in range(1, 5) for s in combinations(range(4), n))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def family(op):
    return "blur_noise" if op.startswith("gaussian_") else "photometric"


def key(labels):
    return ",".join(map(str, labels))


def units_from_panel(panel, train_rows):
    """Validate complete paired views and join to the frozen TRAIN manifest."""
    if not panel or len({p["image"] for p in panel}) != len(panel):
        raise ValueError("Empty or duplicated probe panel")
    expected = {f"{op}/{level}/{draw}" for op, level in ACTIONS
                for draw in range(2 if op == "gaussian_noise" else 1)}
    units = []
    for p in panel:
        row = train_rows.get(p["image"])
        if row is None or row["split"] != "train":
            raise ValueError("Probe not in frozen training manifest")
        n = len(row["targets"])
        if (p["classes"] != [t["class_index"] for t in row["targets"]]
                or p["object_indices"] != [t["object_index"] for t in row["targets"]]
                or p["source_cluster_id"] != row["source_cluster_id"]):
            raise ValueError("Probe object/cluster identity mismatch")
        positions = p["target_positions"]
        if (not positions or len(set(positions)) != len(positions)
                or any(type(i) is not int or not 0 <= i < n for i in positions)):
            raise ValueError("Invalid group positions")
        if set(p["views"]) != expected:
            raise ValueError("Incomplete or unexpected probe views")
        for obs in [p["reference"], *p["views"].values()]:
            values = obs["detected"]
            if len(values) != n or any(type(v) is not bool for v in values):
                raise ValueError("Invalid per-object detection indicators")
        labels = tuple(sorted({p["classes"][i] for i in positions}))
        for i in positions:
            values = [sum(p["views"][f"{op}/{level}/{d}"]["detected"][i]
                          for d in range(2 if op == "gaussian_noise" else 1))
                      / (2 if op == "gaussian_noise" else 1) for op, level in ACTIONS]
            units.append({"image": p["image"], "object_index": p["object_indices"][i],
                          "cluster": p["source_cluster_id"], "source": row["source"],
                          "class": p["classes"][i], "class_set": labels,
                          "reference_detected": p["reference"]["detected"][i],
                          "retained": values})
    return units


def class_stats(units, settings):
    all_units = list(units)
    found = [u for u in all_units if u["reference_detected"]]
    n = len(found)
    clusters = sorted({u["cluster"] for u in found})
    rate = np.mean([u["retained"] for u in found], axis=0) if n else np.zeros(len(ACTIONS))
    support = n >= settings["minimum_detected_objects"] and len(clusters) >= settings["minimum_detected_clusters"]
    allowed = (rate >= settings["retention_threshold"]) & support
    return {"objects": len(all_units), "detected_objects": n,
            "detected_clusters": len(clusters), "supported": support,
            "reference_recall": n/len(all_units) if all_units else None,
            "retention": rate.tolist() if n else [None]*len(ACTIONS),
            "allowed": allowed.tolist()}


def balanced_distribution(allowed):
    """Uniform family -> eligible operation -> eligible tested level."""
    if len(allowed) != len(ACTIONS):
        raise ValueError("Wrong action dimension")
    active = defaultdict(lambda: defaultdict(list))
    for j, ((op, level), yes) in enumerate(zip(ACTIONS, allowed)):
        if yes:
            active[family(op)][op].append(j)
    result = np.zeros(len(ACTIONS) + 1)  # Final coordinate is identity.
    if not active:
        result[-1] = 1.
    else:
        for ops in active.values():
            for indices in ops.values():
                result[indices] = 1/(len(active)*len(ops)*len(indices))
    return result.tolist()


def draw_action(allowed, seed, epoch, image, group):
    """Independent keyed choice; deterministic, no global RNG consumption."""
    probabilities = balanced_distribution(allowed)
    word = int(digest(["v2_action", seed, epoch, image, group])[:16], 16)
    uniform = (word >> 11) / 2**53
    total = 0.
    for j, p in enumerate(probabilities):
        total += p
        if uniform < total:
            return None if j == len(ACTIONS) else ACTIONS[j]
    raise RuntimeError("Invalid probability mass")


def build_policy(units, settings):
    pooled = {str(c): class_stats((u for u in units if u["class"] == c), settings) for c in range(4)}
    direct, veto, routed = {}, {}, {}
    for labels in SETS:
        s = key(labels)
        direct[s] = {str(c): class_stats((u for u in units if u["class_set"] == labels and u["class"] == c), settings)
                     for c in labels}
        bad = np.zeros(len(ACTIONS), dtype=bool)
        if len(labels) > 1:
            for cell in direct[s].values():
                if cell["supported"]:
                    bad |= np.asarray(cell["retention"]) < settings["retention_threshold"]
        veto[s] = bad.tolist()
        raw = np.logical_and.reduce([pooled[str(c)]["allowed"] for c in labels])
        supported_direct = all(cell["supported"] for cell in direct[s].values())
        allowed = raw & ~bad
        routed[s] = {"allowed": allowed.tolist(), "vetoed_actions": int((raw & bad).sum()),
                     "direct_supported": supported_direct,
                     "borrowed_evidence": not supported_direct,
                     "distribution": balanced_distribution(allowed)}
    common = np.logical_and.reduce([cell["allowed"] for cell in pooled.values()])
    common &= ~np.logical_or.reduce(list(veto.values()))
    return {"pooled": pooled, "direct": direct, "routed": routed,
            "shared_allowed": common.tolist(), "shared_distribution": balanced_distribution(common)}


def tv(first, second):
    return float(np.abs(np.asarray(first)-np.asarray(second)).sum()/2)


def stability(units, settings, seed, bootstrap_draws):
    """Cluster sensitivity, separated from formal statistical inference."""
    out = {}
    for c in range(4):
        group = [u for u in units if u["class"] == c]
        cell = class_stats(group, settings)
        found = [u for u in group if u["reference_detected"]]
        clusters = sorted({u["cluster"] for u in found})
        by_cluster = {cluster: [u for u in found if u["cluster"] == cluster] for cluster in clusters}
        counts = np.array([len(v) for v in by_cluster.values()], dtype=np.int64)
        sums = np.array([np.sum([u["retained"] for u in v], axis=0) for v in by_cluster.values()]).reshape(-1, len(ACTIONS))
        n = int(counts.sum())
        base = np.asarray(cell["allowed"])
        flips, rates = [], []
        if len(clusters):
            for j in range(len(clusters)):
                remaining = n-int(counts[j])
                support = remaining >= settings["minimum_detected_objects"] and len(clusters)-1 >= settings["minimum_detected_clusters"]
                rate = (sums.sum(axis=0)-sums[j])/remaining if remaining else np.zeros(len(ACTIONS))
                flips.append(((rate >= settings["retention_threshold"]) & support) != base)
            rng = np.random.default_rng(seed + c)
            for _ in range(bootstrap_draws):
                idx = rng.integers(0, len(clusters), size=len(clusters))
                rates.append(sums[idx].sum(axis=0)/counts[idx].sum() >= settings["retention_threshold"])
        # The split is by all source clusters, including reference misses.
        all_clusters = sorted({u["cluster"] for u in group}, key=lambda v: digest(["halves", seed, v]))
        left = set(all_clusters[::2])
        halves = [class_stats((u for u in group if (u["cluster"] in left) == flag), settings) for flag in (True, False)]
        source_stress = {}
        for source in sorted({u["source"] for u in group}):
            reduced = class_stats((u for u in group if u["source"] != source), settings)
            source_stress[source] = {"supported": reduced["supported"],
                                     "changed_actions": int(np.sum(np.asarray(reduced["allowed"]) != base))}
        out[str(c)] = {"baseline": cell,
                       "leave_one_cluster_changed_actions": np.any(flips, axis=0).tolist() if flips else [False]*len(ACTIONS),
                       "leave_one_cluster_flip_counts": np.sum(flips, axis=0).tolist() if flips else [0]*len(ACTIONS),
                       "cluster_deletions": len(clusters),
                       "conditional_bootstrap_retention_pass_frequency": np.mean(rates, axis=0).tolist() if rates else [None]*len(ACTIONS),
                       "halves": halves,
                       "half_action_agreement": float(np.mean(np.asarray(halves[0]["allowed"]) == halves[1]["allowed"])) if all(h["supported"] for h in halves) else None,
                       "source_withholding": source_stress}
    return out
