"""Recompute the fixed endpoint directly with COCOeval and independent GT assembly."""
from collections import Counter, defaultdict
import contextlib
import copy
import gc
import gzip
from importlib.metadata import version
import io
import json
from pathlib import Path
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from audit_returned import ROOT, REPORT, RETURNED, load, save

CLASSES = ('D00', 'D10', 'D20', 'D40')


def main():
    rows = [json.loads(line) for line in (ROOT/'detection_splits_v1/val.jsonl').read_text().splitlines()]
    assert len(rows) == 5760 and all(r['split'] == 'val' for r in rows)
    truth_data = {'info': {}, 'images': [], 'annotations': [],
                  'categories': [{'id': c+1, 'name': name} for c, name in enumerate(CLASSES)]}
    for image_id, row in enumerate(rows):
        truth_data['images'].append({'id': image_id, 'width': row['width'], 'height': row['height']})
        for item in row['targets']:
            x1, y1, x2, y2 = item['box_xyxy']
            truth_data['annotations'].append({'id': len(truth_data['annotations'])+1, 'image_id': image_id,
                'category_id': item['class_index']+1, 'bbox': [x1, y1, x2-x1, y2-y1],
                'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
    results = {}
    for arm, epoch in [('common_warmup', 1)] + [(a, 4) for a in ('geometry_only','fixed_local','online_shared','class_adaptive')]:
        start = time.monotonic()
        path = RETURNED/arm/f'epoch_{epoch:03d}_val_predictions.json.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as handle: predictions = json.load(handle)
        saved = load(RETURNED/arm/f'epoch_{epoch:03d}_validation.json')['metrics']
        count = Counter()
        for p in predictions:
            i = p['image_id']; assert type(i) is int and 0 <= i < len(rows)
            assert p['category_id'] in (1, 2, 3, 4)
            assert np.isfinite(p['score']) and 0 <= p['score'] <= 1
            x, y, w, h = p['bbox']; assert np.isfinite(p['bbox']).all() and w > 0 and h > 0
            assert x >= -1e-8 and y >= -1e-8 and x+w <= rows[i]['width']+1e-8 and y+h <= rows[i]['height']+1e-8
            count[i] += 1
        assert max(count.values(), default=0) <= 100
        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO(); truth.dataset = copy.deepcopy(truth_data); truth.createIndex()
            detection = truth.loadRes(predictions)
            evaluator = COCOeval(truth, detection, 'bbox')
            evaluator.params.imgIds = list(range(len(rows)))
            evaluator.params.catIds = [1, 2, 3, 4]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate(); evaluator.accumulate()
        def avg(values):
            valid = values[values >= 0]
            return float(valid.mean()) if valid.size else None
        precision = evaluator.eval['precision']
        values = {'AP50_95': avg(precision[:, :, :, 0, -1]), 'AP50': avg(precision[0, :, :, 0, -1]),
                  'AP75': avg(precision[5, :, :, 0, -1]), 'AP_small': avg(precision[:, :, :, 1, -1]),
                  'AP_medium': avg(precision[:, :, :, 2, -1]), 'AP_large': avg(precision[:, :, :, 3, -1]),
                  'AR100': avg(evaluator.eval['recall'][:, :, 0, -1])}
        errors = [abs(v-saved[k]) for k,v in values.items() if v is not None]
        per_class = {}
        for c, name in enumerate(CLASSES):
            per_class[name] = {'AP50_95': avg(precision[:, :, c, 0, -1]), 'AP50': avg(precision[0, :, c, 0, -1]),
                               'AP_small': avg(precision[:, :, c, 1, -1])}
            errors.extend(abs(v-saved['per_class'][name][k]) for k,v in per_class[name].items() if v is not None)
        max_error = max(errors)
        assert max_error < 1e-12, (arm, max_error)
        negative = {i for i,row in enumerate(rows) if not row['targets']}
        negative_counts = Counter(p['image_id'] for p in predictions if p['image_id'] in negative and p['score'] >= .25)
        assert saved['no_target_predicted_boxes'] == {str(i): negative_counts[i] for i in sorted(negative)}
        assert saved['no_target_images_with_predictions'] == len(negative_counts)
        # Independent matching at the declared diagnostic threshold. This is
        # descriptive precision/recall, not a tuned confidence threshold.
        grouped = defaultdict(list)
        for p in predictions:
            if p['score'] >= .25: grouped[p['image_id']].append(p)
        tp = Counter(); fp = Counter(); support = Counter()
        for i,row in enumerate(rows):
            targets = row['targets']; used = set()
            support.update(t['class_index'] for t in targets)
            for p in sorted(grouped[i], key=lambda p: -p['score']):
                cl = p['category_id']-1; x,y,w,h = p['bbox']; choices = []
                for j,t in enumerate(targets):
                    if j in used or t['class_index'] != cl: continue
                    a,b,c,d = t['box_xyxy']
                    intersect = max(0., min(x+w,c)-max(x,a))*max(0., min(y+h,d)-max(y,b))
                    overlap = intersect/(w*h+(c-a)*(d-b)-intersect)
                    if overlap >= .5: choices.append((overlap,-j))
                if choices:
                    _, negj = max(choices); used.add(-negj); tp[cl] += 1
                else: fp[cl] += 1
        threshold = {'confidence': .25, 'iou': .5, 'true_positives': sum(tp.values()), 'false_positives': sum(fp.values()),
                     'target_objects': sum(support.values()), 'recall': sum(tp.values())/sum(support.values()),
                     'precision': sum(tp.values())/(sum(tp.values())+sum(fp.values())),
                     'per_class': {CLASSES[c]: {'tp': tp[c], 'fp': fp[c], 'targets': support[c], 'recall': tp[c]/support[c]} for c in range(4)}}
        results[arm] = {'epoch': epoch, 'metrics': values, 'per_class': per_class,
                       'maximum_metric_error': max_error, 'images': len(rows), 'target_boxes': len(truth_data['annotations']),
                       'prediction_boxes': len(predictions), 'no_target_images': len(negative),
                       'no_target_images_with_predictions': len(negative_counts), 'no_target_boxes': sum(negative_counts.values()),
                       'threshold_diagnostic': threshold, 'seconds': time.monotonic()-start}
        print(f'COCO REPLAY {arm}: PASS max_error={max_error:.3g} AP={values["AP50_95"]*100:.5f} negatives={len(negative_counts)}/{len(negative)} time={time.monotonic()-start:.1f}s', flush=True)
        save('metric_recomputation.json', {'versions': {'numpy': np.__version__, 'pycocotools': version('pycocotools')},
             'scope': 'Official COCOeval from saved predictions and independently constructed GT. No detector re-inference; original-image pixel-area definitions.',
             'results': results})
        del evaluator, truth, detection, predictions, grouped
        gc.collect()


if __name__ == '__main__': main()
