"""Read-only evidence replay on two v1 trajectories; never predicts v2 AP."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import controller as c
from rdd_data import read_manifest


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_records(root, trajectory, epochs):
    folder = root/"reports"/trajectory
    audit_path = folder/"ccast_verification.json"
    verification_path = folder/"controller_verification.json"
    audit = json.loads(audit_path.read_text())
    verified = json.loads(verification_path.read_text())
    protocol_path = root/trajectory/"protocol.json"
    settings = json.loads(protocol_path.read_text())["controller"]
    if (settings["confidence"] != .25 or settings["iou"] != .5
            or settings["noise_draws"] != 2 or len(settings["levels"]) != 3):
        raise ValueError("Archived observer/grid incompatible with replay")
    archive = root/audit["archive"]
    if file_hash(archive) != audit["archive_sha256"]:
        raise ValueError("Historical archive hash mismatch")
    expected = {r["epoch"]: r["policy_record_sha256"] for r in verified["rounds"]}
    if sorted(expected) != epochs:
        raise ValueError("Historical audit round coverage differs")
    records, member_hashes = {}, {}
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if "/controller/" in member.name and member.name.endswith(".json.gz"):
                if not member.isfile() or member.size > 100_000_000:
                    raise ValueError("Unexpected probe member")
                data = tf.extractfile(member).read()
                record = json.loads(gzip.decompress(data))
                if record["settings_sha256"] != c.digest(settings):
                    raise ValueError("Archived observer settings checksum mismatch")
                epoch = record["epoch"]
                actual = hashlib.sha256(data).hexdigest()
                if epoch in records or actual != expected.get(epoch):
                    raise ValueError("Probe provenance mismatch or duplicate epoch")
                records[epoch] = record
                member_hashes[member.name] = actual
                if len(records) == len(expected):
                    break
    if sorted(records) != epochs:
        raise ValueError("Missing probe evidence")
    return records, {"archive": archive.name, "archive_sha256": audit["archive_sha256"],
                     "audit_sha256": file_hash(audit_path), "controller_verification_sha256": file_hash(verification_path),
                     "historical_protocol_sha256": file_hash(protocol_path),
                     "probe_member_sha256": member_hashes}


def summarize(policy, sensitivity, panel):
    counts = Counter(c.key(tuple(sorted({p["classes"][i] for i in p["target_positions"]}))) for p in panel)
    total = sum(counts.values())
    active, borrowed, mixed, mixed_active, different = 0, 0, 0, 0, 0
    distance = 0.
    for labels, count in counts.items():
        route = policy["routed"][labels]
        yes = any(route["allowed"])
        active += count*yes
        borrowed += count*(yes and route["borrowed_evidence"])
        mixed += count*("," in labels)
        mixed_active += count*(yes and "," in labels)
        d = c.tv(route["distribution"], policy["shared_distribution"])
        distance += count*d
        different += count*(d > 1e-12)
    allowed = sum(sum(cell["allowed"]) for cell in policy["pooled"].values())
    vulnerable_allowed = sum(sum(bool(a) and bool(b) for a, b in zip(v["baseline"]["allowed"], v["leave_one_cluster_changed_actions"])) for v in sensitivity.values())
    return {"panel_groups": total, "supported_classes": sum(s["supported"] for s in policy["pooled"].values()),
            "supported_class_names": [str(k) for k, s in policy["pooled"].items() if s["supported"]],
            "shared_actions": sum(policy["shared_allowed"]), "pooled_class_actions": allowed,
            "active_groups": active, "active_mixed_groups": mixed_active, "mixed_groups": mixed,
            "active_groups_borrowing_direct_evidence": borrowed,
            "groups_different_from_shared": different,
            "group_weighted_total_variation_to_shared": distance/total,
            "admissible_class_actions_changed_by_one_cluster": vulnerable_allowed,
            "distinct_single_class_distributions": len({c.digest(policy["routed"][str(i)]["distribution"]) for i in range(4)})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.project.resolve(), args.out.resolve()
    cfg = json.loads(Path(__file__).with_name("protocol.json").read_text())
    rows = read_manifest(root/"detection_splits_v1"/"train.jsonl", cfg["train_manifest_sha256"])
    if len(rows) != cfg["expected_train_images"]:
        raise ValueError("Wrong training row count")
    out.mkdir(parents=True, exist_ok=False)
    save(out/"source_protocol.json", cfg)
    train = {r["image"]: r for r in rows}
    all_summary, sources = [], {}
    for trajectory in cfg["retrospective"]["trajectories"]:
        records, provenance = load_records(root, trajectory, cfg["retrospective"]["epochs"])
        sources[trajectory] = provenance
        for epoch, record in sorted(records.items()):
            units = c.units_from_panel(record["panel"], train)
            policy = c.build_policy(units, cfg["controller"])
            sensitivity = c.stability(units, cfg["controller"], cfg["seed"]+epoch*100,
                                      cfg["retrospective"]["bootstrap_draws"])
            summary = {"trajectory": trajectory, "epoch": epoch, **summarize(policy, sensitivity, record["panel"])}
            all_summary.append(summary)
            save(out/f"{trajectory}_epoch_{epoch:02d}.json", {"policy": policy, "stability": sensitivity, "summary": summary})
            print(f"REPLAY {trajectory} epoch={epoch} supported={summary['supported_classes']}/4 shared_actions={summary['shared_actions']} active_groups={summary['active_groups']}/{summary['panel_groups']} borrowed={summary['active_groups_borrowing_direct_evidence']} unstable_allowed={summary['admissible_class_actions_changed_by_one_cluster']}/{summary['pooled_class_actions']}", flush=True)
    with (out/"summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0]))
        writer.writeheader(); writer.writerows(all_summary)
    save(out/"summary.json", {"status": "retrospective_replay_complete_not_efficacy", "rounds": all_summary,
                             "sources": sources, "protocol_sha256": file_hash(Path(__file__).with_name("protocol.json")),
                             "source_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                             "limitations": cfg["retrospective"]["not_identified"]})
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
