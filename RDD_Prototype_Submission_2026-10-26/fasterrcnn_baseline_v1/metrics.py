"""COCO bbox metrics on original stored-image coordinates, including empty images."""
from collections import Counter
import contextlib
import copy
import io
import math

import numpy as np

from integration import prediction_summary, require
from rdd_data import CLASSES, restore_boxes


def ground_truth(examples):
    images, annotations, seen = [], [], set()
    for example in examples:
        row, image_id = example.record, example.image_id
        require(image_id not in seen, "Duplicate evaluation image")
        seen.add(image_id)
        images.append({"id": image_id, "file_name": row["image"],
                       "width": row["width"], "height": row["height"]})
        for target in row["targets"]:
            x1, y1, x2, y2 = target["box_xyxy"]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": target["class_index"] + 1,
                                "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    return {"info": {"description": "RDD detection ground truth; partition recorded by caller"},
            "images": images, "annotations": annotations,
            "categories": [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]}


def convert_prediction(example, prediction):
    boxes, scores, labels = (prediction[k] for k in ("boxes", "scores", "labels"))
    prediction_summary(boxes, scores, labels, 0)
    restored = restore_boxes(boxes.numpy(), example.geometry)
    width, height = example.record["width"], example.record["height"]
    clipped = restored.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    valid = (clipped[:, 2:] > clipped[:, :2]).all(axis=1)
    results = []
    for box, score, label in zip(clipped[valid], scores.numpy()[valid], labels.numpy()[valid]):
        x1, y1, x2, y2 = box.tolist()
        results.append({"image_id": example.image_id, "category_id": int(label) + 1,
                        "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
    return results, {"before_clipping": len(boxes), "after_clipping": len(results),
                     "boxes_clipped": int(np.any(clipped != restored, axis=1).sum()),
                     "collapsed_after_clipping": int((~valid).sum())}


def evaluate(gt, predictions, negative_confidence=0.25):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    image_ids = [i["id"] for i in gt["images"]]
    require(len(set(image_ids)) == len(image_ids), "Duplicate evaluation image ID")
    require(len({a["id"] for a in gt["annotations"]}) == len(gt["annotations"]),
            "Duplicate annotation ID")
    image_set = set(image_ids)
    for p in predictions:
        require(p["image_id"] in image_set and type(p["category_id"]) is int
                and 1 <= p["category_id"] <= 4, "Unknown prediction image or class")
        require(math.isfinite(p["score"]) and 0 <= p["score"] <= 1, "Invalid score")
        require(len(p["bbox"]) == 4 and all(math.isfinite(v) for v in p["bbox"])
                and min(p["bbox"][2:]) > 0, "Invalid COCO prediction box")
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(gt)
        truth.createIndex()
        if predictions:
            detections = truth.loadRes(copy.deepcopy(predictions))
        else:
            detections = COCO()
            detections.dataset = {"images": copy.deepcopy(gt["images"]),
                                  "categories": copy.deepcopy(gt["categories"]), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(truth, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = [1, 2, 3, 4]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]

    def mean_valid(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None

    per_class = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = {"AP50_95": mean_valid(precision[:, :, index, 0, -1]),
                           "AP50": mean_valid(precision[0, :, index, 0, -1]),
                           "target_count": sum(a["category_id"] == index+1 for a in gt["annotations"])}
    positive_ids = {a["image_id"] for a in gt["annotations"]}
    negative_ids = image_set - positive_ids
    negative_totals = Counter(p["image_id"] for p in predictions if p["score"] >= negative_confidence)
    negative_counts = {str(i): negative_totals[i] for i in sorted(negative_ids)}
    return {"AP50_95": mean_valid(precision[:, :, :, 0, -1]),
            "AP50": mean_valid(precision[0, :, :, 0, -1]),
            "AP75": mean_valid(precision[5, :, :, 0, -1]),
            "AR100": mean_valid(evaluator.eval["recall"][:, :, 0, -1]),
            "per_class": per_class, "evaluation_images": len(image_ids),
            "target_boxes": len(gt["annotations"]), "prediction_boxes": len(predictions),
            "no_target_images": len(negative_ids), "negative_confidence": negative_confidence,
            "no_target_predicted_boxes": negative_counts,
            "no_target_images_with_predictions": sum(count > 0 for count in negative_counts.values()),
            "scope": "partition recorded by caller; undefined metrics are null; all retained targets are ordinary ground truth"}
