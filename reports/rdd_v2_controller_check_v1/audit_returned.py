"""Audit saved controller evidence without new inference or historical edits."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PACKAGE = ROOT/"rdd_v2_controller_check_v1"
ARCHIVE = ROOT/"rdd_v2_controller_check_v1_results.tar.gz"
EXPECTED = "333154238321e50b2b32dc94710a55ffe0480d53096cded9b57114da14d3c2ae"
sys.path.insert(0, str(PACKAGE))
import controller
import diagnostic
import geometry
import local_augmentation as local


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if path.exists():
        assert path.read_bytes() == data, f"Existing audit differs: {path}"
    else:
        with path.open("xb") as handle:
            handle.write(data)


def extract():
    assert sha(ARCHIVE) == EXPECTED
    folder = OUT/"returned_records"
    folder.mkdir(exist_ok=True)
    files = {}
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert len(members) < 2048 and sum(m.size for m in members) < 1_000_000_000
        for m in members:
            name = PurePosixPath(m.name)
            assert not name.is_absolute() and ".." not in name.parts and "\\" not in m.name and ":" not in m.name
            assert name.parts[0] == "rdd_v2_controller_check_v1"
            assert m.isfile() or m.isdir()
            if m.isdir():
                continue
            assert m.size < 40_000_000
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            path = folder/relative
            assert path.resolve().is_relative_to(folder.resolve())
            data = tar.extractfile(m).read()
            files[relative] = hashlib.sha256(data).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data
            else:
                with path.open("xb") as stream:
                    stream.write(data)
    checks = {}
    for line in (folder/"SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in checks
        checks[name] = digest
    assert checks == {k: v for k, v in files.items() if k != "SHA256SUMS"}
    return folder, checks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same(actual, expected, path="value"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k, v in expected.items():
            same(actual[k], v, path+"."+str(k))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, v in enumerate(expected):
            same(actual[i], v, path+f"[{i}]")
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=0, atol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def match(prediction, gt, labels):
    """Independent scalar candidate matching; no production metric function."""
    boxes = np.asarray(prediction["boxes"], float).reshape(-1, 4)
    scores = np.asarray(prediction["scores"], float)
    cats = np.asarray(prediction["labels"])
    assert scores.shape == cats.shape == (len(boxes),)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((boxes >= 0) & (boxes <= 640)).all() and (boxes[:, 2:] > boxes[:, :2]).all()
    assert ((cats >= 0) & (cats < 4) & (cats == cats.astype(int))).all()
    order = sorted((i for i, score in enumerate(scores) if score >= .25), key=lambda i: (-scores[i], i))
    detected, ious, fp = [False]*len(gt), [0.]*len(gt), 0
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        candidates = []
        for j, (a, b, c, d) in enumerate(gt):
            if detected[j] or labels[j] != cats[i]:
                continue
            intersection = max(0., min(x2, c)-max(x1, a))*max(0., min(y2, d)-max(y1, b))
            overlap = intersection/((x2-x1)*(y2-y1)+(c-a)*(d-b)-intersection)
            if overlap >= .5:
                candidates.append((float(overlap), -j))
        if candidates:
            overlap, minus_j = max(candidates)
            detected[-minus_j], ious[-minus_j] = True, overlap
        else:
            fp += 1
    return {"detected": detected, "iou": ious, "false_positives": fp, "confident_predictions": len(order)}


def independent_gate(units):
    found = [u for u in units if u["reference_detected"]]
    clusters = {u["cluster"] for u in found}
    n = len(found)
    rates = [sum(u["retained"][j] for u in found)/n for j in range(24)] if n else [None]*24
    support = n >= 16 and len(clusters) >= 8
    return {"objects": len(units), "detected_objects": n, "detected_clusters": len(clusters),
            "supported": support, "reference_recall": n/len(units) if units else None,
            "retention": rates, "allowed": [bool(support and r >= .8) for r in rates] if n else [False]*24}


def distribution(allowed):
    groups = defaultdict(lambda: defaultdict(list))
    for j, ((op, level), yes) in enumerate(zip(controller.ACTIONS, allowed)):
        if yes:
            groups["quality" if op.startswith("gaussian_") else "intensity"][op].append(j)
    weights = [0.]*25
    if not groups:
        weights[-1] = 1.
    for family in groups.values():
        for items in family.values():
            for j in items:
                weights[j] = 1/len(groups)/len(family)/len(items)
    return weights


def audit_policy(units, policy):
    pooled = {str(c): independent_gate([u for u in units if u["class"] == c]) for c in range(4)}
    same(policy["pooled"], pooled)
    vetoes = []
    for labels in controller.SETS:
        key = controller.key(labels)
        direct = {str(c): independent_gate([u for u in units if u["class"] == c and u["class_set"] == labels]) for c in labels}
        same(policy["direct"][key], direct)
        veto = [any(len(labels)>1 and cell["supported"] and cell["retention"][j]<.8 for cell in direct.values()) for j in range(24)]
        vetoes.append(veto)
        raw = [all(pooled[str(c)]["allowed"][j] for c in labels) for j in range(24)]
        allowed = [yes and not bad for yes, bad in zip(raw, veto)]
        expected = {"allowed": allowed, "vetoed_actions": sum(yes and bad for yes, bad in zip(raw, veto)),
                    "direct_supported": all(v["supported"] for v in direct.values()),
                    "borrowed_evidence": not all(v["supported"] for v in direct.values()),
                    "distribution": distribution(allowed)}
        same(policy["routed"][key], expected)
    common = [all(v["allowed"][j] for v in pooled.values()) and not any(v[j] for v in vetoes) for j in range(24)]
    same(policy["shared_allowed"], common)
    same(policy["shared_distribution"], distribution(common))


def main():
    folder, checks = extract()
    cfg, result, context, plan = [read(folder/name) for name in ("protocol.json", "result.json", "context.json", "panel_plan.json")]
    assert cfg == read(PACKAGE/"protocol.json") and plan == read(PACKAGE/"panel_plan.json")
    assert context["package_hashes"] == diagnostic.verify_package(PACKAGE)
    assert context["versions"] == cfg["versions"] and context["checkpoint"] == cfg["checkpoint"]
    manifest = ROOT/"detection_splits_v1/train.jsonl"
    assert sha(manifest) == cfg["train_manifest_sha256"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == cfg["expected_train_images"] and all(r["split"] == "train" for r in rows)
    by_image = {r["image"]: r for r in rows}
    assert plan == diagnostic.plan_panels(rows, cfg)
    # The separate geometry audit uses homogeneous corner matrices, independently
    # of the production forward_boxes function reused by the experiment.
    spec = importlib.util.spec_from_file_location("independent_geometry_audit", ROOT/"reports/rdd_v2_geometry_check_v1/audit_returned.py")
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    panels = {name: [] for name in ("A", "B")}
    source_geometry, op_changes, geometry_events = defaultdict(Counter), defaultdict(Counter), []
    max_error, n_match, n_predictions, n_annotations, crop_identity = 0., 0, 0, 0, 0
    for panel in ("A", "B"):
        for index, selected in enumerate(plan["panels"][panel], 1):
            p = read(folder/"observations"/f"{panel}_{index:03d}.json")
            assert p["selection"] == selected and p["image"] == selected["image"]
            row = by_image[p["image"]]
            assert p["target_positions"] == selected["target_positions"]
            assert p["object_indices"] == [t["object_index"] for t in row["targets"]]
            assert p["classes"] == [t["class_index"] for t in row["targets"]]
            assert p["source_cluster_id"] == row["source_cluster_id"]
            annotations, prepared_corners = geom.prepare(row, 640)
            clean_boxes = geom.rectangles(prepared_corners)
            candidate = p["geometry"]["candidate"]
            assert candidate == geometry.chosen_candidate(row, cfg["geometry"], cfg["seed"], cfg["geometry_epoch_index"])
            s, left, top = candidate["scale"], candidate["left"], candidate["top"]
            transform = np.array([[-s if candidate["flip"] else s, 0, left+(640*s if candidate["flip"] else 0)],
                                  [0, s, top], [0, 0, 1]], float)
            transformed = geom.rectangles(prepared_corners @ transform.T)
            cropped = [a["id"] for a, box in zip(annotations, transformed) if min(box[:2])<0 or max(box[2:])>640]
            assert p["geometry"]["accepted"] == (not cropped) and p["geometry"]["cropped_ids"] == cropped
            mapped = clean_boxes if cropped else transformed
            source_geometry[row["source"]]["fallback" if cropped else "accepted"] += 1
            returned_ann = p["reference"]["render"]["annotations"]
            assert len(returned_ann) == len(annotations)
            for ann, recorded, box in zip(annotations, returned_ann, mapped):
                assert {k: recorded[k] for k in ann} == ann
                error = float(np.abs(box-recorded["box_xyxy"]).max())
                assert error < 1e-10
                max_error = max(max_error, error)
                n_annotations += 1
            if cropped or (not candidate["flip"] and s == 1 and left == top == 0):
                assert p["clean"]["pixel_sha256"] == p["reference"]["pixel_sha256"]
                crop_identity += 1
            all_views = {"clean": p["clean"], "reference": p["reference"], **p["views"]}
            assert len(all_views) == 29
            for name, obs in all_views.items():
                expected_boxes = (clean_boxes if name == "clean" else mapped)[:len(row["targets"])]
                actual_boxes = np.asarray(obs["render"]["boxes"])
                assert np.allclose(actual_boxes, expected_boxes, atol=1e-10, rtol=0)
                assert obs["render"]["labels"] == p["classes"]
                independent = match(obs["prediction"], actual_boxes, p["classes"])
                same({k: obs[k] for k in independent}, independent)
                n_match += 1
                n_predictions += len(obs["prediction"]["scores"])
                assert len(obs["pixel_sha256"]) == 64 and all(x in "0123456789abcdef" for x in obs["pixel_sha256"])
                if name not in ("clean", "reference"):
                    operation, level, draw = name.split("/")
                    decision = obs["render"]["decision"]
                    assert decision["operation"] == operation and decision["parameters"] == diagnostic.parameters(operation, int(level))
                    assert decision["image"] == row["image"]
                    expected_decision = local.attach_local(row, {k:v for k,v in decision.items() if k != "local"}, cfg["seed"], 0, 640, selected["group_index"])
                    assert expected_decision == decision
                    assert decision["local"]["object_indices"] == selected["object_indices"]
                    seed = int(controller.digest(["v2_probe_noise", cfg["seed"], row["image"], int(level), int(draw)])[:16], 16)
                    assert decision["noise_seed"] == (seed if operation == "gaussian_noise" else None)
                    changed = obs["render"]["changed_pixels_vs_reference"]
                    assert bool(changed) == (obs["pixel_sha256"] != p["reference"]["pixel_sha256"])
                    before = obs["render"]["changed_pixels_before_geometry"]
                    assert 0 <= before <= decision["local"]["union_pixels"] and 0 <= changed <= 640**2
                    op_changes[operation]["views"] += 1
                    op_changes[operation]["unchanged_before"] += int(before == 0)
                    op_changes[operation]["unchanged_after"] += int(changed == 0)
            for position in p["target_positions"]:
                a, b = p["clean"]["detected"][position], p["reference"]["detected"][position]
                if a != b:
                    geometry_events.append({"panel": panel, "image": p["image"], "source": row["source"],
                        "class": p["classes"][position], "object_index": p["object_indices"][position],
                        "event": "lost" if a else "found", "candidate": candidate,
                        "clean_box": p["clean"]["render"]["boxes"][position],
                        "reference_box": p["reference"]["render"]["boxes"][position]})
            panels[panel].append(p)
    assert len(list((folder/"observations").glob("*.json"))) == 512
    reproduced = diagnostic.analyze(panels, rows, cfg)
    same({k: result[k] for k in reproduced}, reproduced)
    for name in panels:
        units = controller.units_from_panel(panels[name], by_image)
        audit_policy(units, result["panels"][name]["policy"])
    assert result["detector_images"] == 512*29+4 == cfg["max_detector_images"]
    assert result["forward_calls"] == 512*15+2
    assert result["detector_audit"]["unchanged"]
    assert result["detector_audit"]["model_sha256_before_after"] == context["checkpoint_model_sha256"]
    assert result["parity"]["max_abs_error"] == 0 and result["elapsed_seconds"] < cfg["administrative_seconds"]
    assert all(len({s["image"] for s in plan["panels"][n]}) == 256 for n in panels)
    all_selection = plan["panels"]["A"] + plan["panels"]["B"]
    assert len({p["source_cluster_id"] for p in all_selection}) == 512
    assert all(Counter(s["anchor_class"] for s in plan["panels"][n]) == {0:64,1:64,2:64,3:64} for n in panels)
    summary = {"archive_sha256": EXPECTED, "internal_files_verified": len(checks),
        "package_files_verified": len(context["package_hashes"]), "frozen_plan_reproduced": True,
        "images": 512, "distinct_clusters": 512, "saved_view_matches_independently_checked": n_match,
        "saved_predictions_checked": n_predictions, "annotations_independently_mapped": n_annotations,
        "max_box_coordinate_error": max_error, "clean_reference_identical_geometry_cases": crop_identity,
        "all_summary_fields_reproduced": True, "pooled_direct_routed_policies_independently_checked": True,
        "source_geometry": dict(source_geometry), "appearance_pixel_changes": dict(op_changes),
        "geometry_detection_changes": geometry_events,
        "reported_invariants": result["detector_audit"], "reported_parity": result["parity"],
        "elapsed_seconds": result["elapsed_seconds"], "forward_seconds": result["forward_seconds"],
        "limitations": ["No local GPU re-inference or remote checkpoint reloading",
                        "Raw 512 training RGB images remain on CCAST; their stored hashes and loader checks were verified through execution records/code",
                        "Visual scope is the returned example sheets and separate thin-case pixels, not all candidate images",
                        "Remote unit-test stdout is absent from the archive; the checked wrapper runs tests with fail-on-error before this job"]}
    save(OUT/"numerical_audit.json", summary)
    images = []
    for path in sorted(folder.rglob("*.png")):
        with Image.open(path) as image:
            image.load()
            images.append({"file":path.relative_to(folder).as_posix(),"size":list(image.size),"mode":image.mode})
    save(OUT/"image_inventory.json", images)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("geometry_detection_changes","source_geometry","appearance_pixel_changes")}, indent=2))


if __name__ == "__main__":
    main()
