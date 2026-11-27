"""Descriptive follow-up to the verified, frozen two-panel diagnostic.

Transfer summaries and attribution of geometry losses are exploratory analyses
of existing records, not new significance tests or new controller thresholds.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

from audit_returned import OUT, ROOT, PACKAGE, save, read
sys.path.insert(0, str(PACKAGE))
import controller
import geometry
import local_augmentation as local


def iou(box, predictions):
    candidates = np.asarray(predictions, dtype=float).reshape(-1, 4)
    a = np.maximum(candidates[:, :2], np.asarray(box)[:2])
    b = np.minimum(candidates[:, 2:], np.asarray(box)[2:])
    intersection = np.maximum(b-a, 0).prod(axis=1)
    union = (candidates[:, 2:]-candidates[:, :2]).prod(axis=1)+(box[2]-box[0])*(box[3]-box[1])-intersection
    return intersection/union


def transformed_support(rectangles, candidate, accepted):
    mask = np.zeros((640, 640), dtype=np.uint8)
    for x1,y1,x2,y2 in rectangles:
        mask[y1:y2,x1:x2] = 255
    if accepted:
        if candidate["flip"]:
            mask = mask[:, ::-1].copy()
        m = candidate["resized_size"]
        resized = np.asarray(Image.fromarray(mask).resize((m,m), Image.Resampling.BILINEAR))
        canvas = np.zeros_like(mask)
        left, top = candidate["left"], candidate["top"]
        x1,y1,x2,y2 = max(left,0),max(top,0),min(left+m,640),min(top+m,640)
        canvas[y1:y2,x1:x2] = resized[y1-top:y2-top,x1-left:x2-left]
        mask = canvas
    # Conservative one-pixel halo allows rounding at an interpolation boundary.
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(3))) > 0


def main():
    folder = OUT/"returned_records"
    result = read(folder/"result.json")
    records = {name:[read(p) for p in sorted((folder/"observations").glob(name+"_*.json"))] for name in ("A","B")}
    rows = {r["image"]:r for r in map(json.loads,(ROOT/"detection_splits_v1/train.jsonl").read_text().splitlines())}
    panels = result["panels"]
    data = {"per_class":{},"panel_diagnostics":{},"geometry_class_summary":{},"preview_checks":[]}
    for c,label in enumerate(("D00","D10","D20","D40")):
        a,b = [panels[n]["policy"]["pooled"][str(c)] for n in ("A","B")]
        differences = []
        for index,action in enumerate(controller.ACTIONS):
            if a["allowed"][index] != b["allowed"][index]:
                differences.append({"action":list(action),"retention_A":a["retention"][index],
                    "retention_B":b["retention"][index],"allowed_A":a["allowed"][index],"allowed_B":b["allowed"][index]})
        transfer = {}
        for source,target in (("A","B"),("B","A")):
            src = panels[source]["policy"]["pooled"][str(c)]
            dst = panels[target]["policy"]["pooled"][str(c)]
            probability = controller.balanced_distribution(src["allowed"])[:24]
            transfer[source+"_policy_on_"+target] = {"admitted_actions":sum(src["allowed"]),
                "below_08_on_other_panel":sum(a and r < .8 for a,r in zip(src["allowed"],dst["retention"])),
                "expected_retention_conditional_on_reference_detections":float(np.dot(probability,dst["retention"]))}
        data["per_class"][label] = {"selected_objects_A_B":[a["objects"],b["objects"]],
            "detected_objects_A_B":[a["detected_objects"],b["detected_objects"]],
            "detected_clusters_A_B":[a["detected_clusters"],b["detected_clusters"]],
            "allowed_A_B":[sum(a["allowed"]),sum(b["allowed"])],
            "agreement":1-len(differences)/24,"differences":differences,"cross_panel_transfer":transfer}
        data["geometry_class_summary"][label] = {
            key:sum(panels[n]["geometry_recall_diagnostic"][str(c)][key] for n in ("A","B"))
            for key in ("selected_objects","clean_detected","geometry_detected","clean_found_geometry_missed","clean_missed_geometry_found")}
    for name,panel in panels.items():
        units = controller.units_from_panel(records[name], rows)
        source_support = {}
        for c in range(4):
            source_support[str(c)] = dict(Counter(u["source"] for u in units if u["class"]==c and u["reference_detected"]))
        data["panel_diagnostics"][name] = {"reference_detected_by_source":source_support,
            "mixed_groups":sum(x["selected_groups"] for x in panel["routing"] if "," in x["class_set"]),
            "borrowed_mixed_groups":sum(x["selected_groups"] for x in panel["routing"] if "," in x["class_set"] and x["borrowed_evidence"]),
            "borrowed_all_groups":sum(x["selected_groups"] for x in panel["routing"] if x["borrowed_evidence"]),
            "groups_different_from_shared":sum(x["selected_groups"] for x in panel["routing"] if x["tv_to_shared"]>0),
            "shared_actions":[list(action) for action,yes in zip(controller.ACTIONS,panel["policy"]["shared_allowed"]) if yes],
            "admitted_actions_loco_sensitive":{c:[list(action) for action,yes,risky in zip(controller.ACTIONS,s["baseline"]["allowed"],s["leave_one_cluster_changed_actions"]) if yes and risky] for c,s in panel["sensitivity"].items()}}
    # Geometry-related D40 misses: confidence and localization diagnostics from
    # all saved post-NMS proposals. This cannot identify why the model changed.
    lost = []
    for name,panel in records.items():
        for p in panel:
            for i in p["target_positions"]:
                if p["classes"][i] != 3 or not p["clean"]["detected"][i] or p["reference"]["detected"][i]:
                    continue
                pr = p["reference"]["prediction"]
                gt = p["reference"]["render"]["boxes"][i]
                overlaps = iou(gt, pr["boxes"])
                correct = [j for j,c in enumerate(pr["labels"]) if c==3]
                confident = [j for j in correct if pr["scores"][j]>=.25]
                localized = [j for j in correct if overlaps[j]>=.5]
                lost.append({"panel":name,"image":p["image"],"object_index":p["object_indices"][i],
                    "best_D40_iou_at_conf025":max((float(overlaps[j]) for j in confident),default=None),
                    "highest_D40_conf_at_iou05":max((pr["scores"][j] for j in localized),default=None),
                    "candidate":p["geometry"]["candidate"]})
    data["D40_geometry_miss_diagnostics"] = lost
    # Existing preview pixels only; no new artwork or raw-image retrieval.
    by_image = {p["image"]:p for panel in records.values() for p in panel}
    keys=("brightness_down/3/0","gaussian_blur/3/0","gaussian_noise/3/0")
    for preview in read(folder/"preview_index.json"):
        p = by_image[preview["image"]]
        with Image.open(folder/preview["file"]) as im:
            arr=np.array(im.convert("RGB"))
        reference=arr[:640,:640]
        local_decision=p["views"][keys[0]]["render"]["decision"]["local"]
        support=transformed_support(local_decision["rectangles"],p["geometry"]["candidate"],p["geometry"]["accepted"])
        # Each panel's 20-pixel title bar differs intentionally and hides pixels.
        support[:20]=True
        counts={}
        for j,key in enumerate(keys,1):
            tile=arr[(j//2)*640:(j//2+1)*640,(j%2)*640:(j%2+1)*640]
            delta=np.any(tile!=reference,axis=2)
            outside=int(np.count_nonzero(delta & ~support))
            assert outside==0,(preview["file"],key,outside)
            counts[key]={"outside_conservative_mask_changes":outside,
                         "admitted_for_this_group_in_its_panel":panels[preview["file"].split("_")[1]]["policy"]["routed"][controller.key(sorted({p["classes"][i] for i in p["target_positions"]}))]["allowed"][list(controller.ACTIONS).index((key.split("/")[0],3))]}
        data["preview_checks"].append({"file":preview["file"],"image":p["image"],"changes":counts})
    thin=read(folder/"thin_case/record.json")
    before=next(a for a in thin["prepared_annotations"] if a["id"]==12)
    after=next(a for a in thin["transformed_annotations"] if a["id"]==12)
    with Image.open(folder/"thin_case/prepared_rgb.png") as im:
        rgb=np.array(im)
        resized=np.array(im.resize((608,608),Image.Resampling.BILINEAR))
    expected=np.full_like(rgb,114)
    expected[16:624,16:624]=resized
    with Image.open(folder/"thin_case/scale095_rgb.png") as im:
        actual=np.array(im)
    error=np.abs(actual.astype(int)-expected.astype(int))
    assert not error.any()
    data["thin_case"]={"prepared_box":before["box_xyxy"],"scaled_box":after["box_xyxy"],
        "prepared_height":before["box_xyxy"][3]-before["box_xyxy"][1],
        "scaled_height":after["box_xyxy"][3]-after["box_xyxy"][1],
        "recomputed_scaled_rgb_max_error":int(error.max()),
        "interpretation":"Positive-area box retained; neither the original 640 view nor a slightly smaller view guarantees learnable detail for a roughly one-pixel-high target"}
    save(OUT/"follow_up_analysis.json",data)
    print(json.dumps({k:v for k,v in data.items() if k not in ("D40_geometry_miss_diagnostics","preview_checks")},indent=2))


if __name__=="__main__":
    main()
