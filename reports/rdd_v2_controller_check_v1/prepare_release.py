"""Create a new diagnostic release, preserving all historical inputs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT/"rdd_v2_controller_check_v1"
REPORT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def hash_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare():
    for name in ("controller.py", "geometry.py", "rdd_data.py"):
        shutil.copyfile(PROJECT/"rdd_v2_preflight_v1"/name, PACKAGE/name)
    for name in ("local_augmentation.py", "shared_augmentation.py"):
        shutil.copyfile(PROJECT/"fasterrcnn_class_adaptive_v1"/name, PACKAGE/name)
    old = json.loads((PROJECT/"rdd_v2_preflight_v1/protocol.json").read_text())
    cfg = {"version": "rdd_v2_controller_check_v1", "seed": 42,
        "purpose": "Fresh training-only support and two-panel policy sensitivity at one fixed detector; not training utility or final efficacy",
        "expected_train_images": old["expected_train_images"], "train_manifest_sha256": old["train_manifest_sha256"],
        "geometry": old["geometry"], "controller": old["controller"],
        "checkpoint": {"file": "epoch_001_step_00006716_04dccd11bbd2.pt", "bytes": 331117946,
            "sha256": "dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858",
            "relative_path": "runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt",
            "selection": "Fixed first unaugmented epoch; not the validation-selected best checkpoint",
            "metadata_source_archive": "fasterrcnn_identity_seed42_v1_records_b07d6daa294f.tar.gz",
            "metadata_source_archive_sha256": "4d16ae384cc8f8663b47a00f84df331ea32c44f3742aebaa71e114b55fc0026f"},
        "panels": {"count": 2, "images_each": 256, "primary_groups_per_class": 64,
            "selection": "Hash-rank target-bearing training images within anchor class; rare classes first; alternate A/B slots; one source cluster across both panels; choose hash-ranked eligible overlap group",
            "rationale": "Resource-bounded enrichment to investigate earlier D40 support failures. 64 anchor groups are not 64 detectable potholes and do not guarantee the 16-object/8-cluster gate.",
            "limitations": "Cluster disjointness removes known duplicate cluster reuse, not all visual/route dependence. Different images can differ in source and difficulty; report composition. Only two panels and one model snapshot; exploratory development design, not a population precision claim.",
            "no_adaptive_resampling": True},
        "pipeline_order": "Existing 640 letterbox -> existing v1 local appearance union -> same whole-image geometry for every paired view. Clean and geometry-only views identify geometry-related misses. Do not substitute geometry-before-local in the later pilot without another renderer check.",
        "geometry_epoch_index": 1, "views_per_image": 29,
        "views": "clean + geometry-only reference + 24 appearance actions with two noise draws at each of its three levels (27 candidate draws)",
        "max_detector_images": 512*29+4, "parity_tolerance": 1e-5,
        "parity": "Repeat the same two-image geometry-reference batch once, four extra image forwards. Check labels/count/order exactly and boxes/scores within absolute 1e-5.",
        "batch_size": 2, "cpu_threads": 4, "bootstrap_draws": 256,
        "administrative_seconds": 3000,
        "resources": "One A10 GPU, 4 CPUs, 32 GB RAM, one hour PBS; 50-minute in-program administrative cap including input/model setup. Stop without a complete archive if unfinished; no automatic retry or full training.",
        "expected_environment": "/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/.venvs/rdd_detection_v1",
        "versions": {"torch": "2.3.0", "torchvision": "0.18.0", "numpy": "1.26.4", "Pillow": "10.3.0"},
        "evaluation": {"score_threshold": .001, "nms_iou": .5, "max_detections": 100},
        "observer": "Ordinary Faster R-CNN predictions, no GT proposals or target argument. One-to-one class-correct matching after inference at score >=0.25 and IoU >=0.5. No optimizer, loss or backward. FP counts are local panel diagnostics, not population FPR.",
        "precision": "float32; AMP/TF32 off; cudnn benchmark off, cudnn deterministic on; measured repeat parity, no general promise of CUDA bitwise training reproducibility",
        "thin_case": {"image": "Norway/train/images/Norway_005967.jpg", "object_index": 12,
                      "purpose": "Additional raw crop, prepared RGB and 0.95-scale QA; no extra detector forwards or insertion into panel selection"},
        "analysis": ["Separate support/recall/allowed actions for A and B; never pool panels to rescue support",
                     "Clean vs geometry recall, including objects lost/found, from selected groups",
                     "Per-class cross-panel allowed-action disagreement and probability TV; unsupported agreement is undefined",
                     "Shared vs class-specific policy difference on the same observations",
                     "Exact mixed-set support/borrowed flags, LOCO sensitivity, cluster halves, conditional bootstrap and source withholding",
                     "Record changed pixels before/after geometry; all rendered views counted, unchanged candidate draws retained"],
        "continuation": {"technical": "Verified inputs/checkpoint, exact forward budget, finite valid predictions, frozen model/gradients/RNG, repeat parity, manual preview review",
                         "scientific": "Review both-panel support, class differentiation and panel/cluster sensitivity. No scalar stability threshold is claimed validated. Unsupported or collapsed policies require a design decision before a matched pilot, not threshold relaxation. Completion is not an automatic go signal.",
                         "next_stage": "Only after review: freeze a bounded four-arm training pilot from a common warmup state; do not start a 20-epoch run from this package"},
        "deferred": {"copy_paste": "disabled", "new_blending_mask": "disabled; exact existing renderer retained",
                     "fixed_roi_auxiliary_loss": "not implemented or used", "validation_test": "not read",
                     "training": "not implemented"}}
    cfg["geometry"]["order"] = "After letterbox and optional existing local appearance: optional reflection, isotropic canvas scaling, integer translation; geometry shared by each paired reference/action set"
    cfg["controller"]["size"] = 640
    save(PACKAGE/"protocol.json", cfg)
    sys.path.insert(0, str(PACKAGE))
    import diagnostic
    import rdd_data
    rows = rdd_data.read_manifest(PROJECT/"detection_splits_v1/train.jsonl", cfg["train_manifest_sha256"])
    plan = diagnostic.plan_panels(rows, cfg)
    save(PACKAGE/"panel_plan.json", plan)
    print(json.dumps({"images": {k: len(v) for k, v in plan["panels"].items()},
                      "distinct_clusters": plan["distinct_clusters"], "sources": plan["sources"],
                      "thin_case_in_panels": any(p["image"] == cfg["thin_case"]["image"] for v in plan["panels"].values() for p in v)}, indent=2))


def release():
    # Run after final test evidence has been written, not before tests.
    validation = json.loads((PACKAGE/"LOCAL_VALIDATION.json").read_text())
    assert validation["status"] == "local_tests_passed_gpu_pending"
    files = sorted(p for p in PACKAGE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "SHA256SUMS")
    assert all(p.suffix in (".py", ".json", ".sh", ".pbs", ".md") for p in files)
    (PACKAGE/"SHA256SUMS").write_text("".join(f"{hash_file(p)}  {p.relative_to(PACKAGE).as_posix()}\n" for p in files), encoding="utf-8", newline="\n")
    archive = PROJECT/"rdd_v2_controller_check_v1.tar.gz"
    assert not archive.exists(), "Refusing to replace an existing release archive"
    with tarfile.open(archive, "x:gz") as tar:
        for path in [*files, PACKAGE/"SHA256SUMS"]:
            tar.add(path, arcname=f"{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}")
    result = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": hash_file(archive), "package_files": len(files)+1}
    save(REPORT/"release.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "release": release}[sys.argv[1]]()
