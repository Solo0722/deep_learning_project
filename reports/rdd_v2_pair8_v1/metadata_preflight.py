"""Check every pilot training draw from frozen annotations; no pixels/evaluation."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'rdd_v2_pair8_v1'))
import numpy as np
import augmentation
import controller
import diagnostic
import geometry
import rdd_data


def main():
    cfg = json.loads((ROOT/'rdd_v2_pair8_v1/protocol.json').read_text())
    rows = rdd_data.read_manifest(ROOT/'detection_splits_v1/train.jsonl', cfg['manifests']['train'])
    plans = json.loads((ROOT/'rdd_v2_pair8_v1/panel_plan.json').read_text())
    lookup = {r['image']: r for r in rows}
    panels = {}
    for epoch, plan in plans.items():
        for selection in plan: geometry.prepared_annotations(lookup[selection['image']], 640)
        panels[epoch] = {'images': len(plan), 'clusters': len({s['source_cluster_id'] for s in plan}),
                         'anchor_classes': dict(Counter(s['anchor_class'] for s in plan)),
                         'sources': dict(Counter(s['source'] for s in plan))}
    result = {'train_manifest_sha256': cfg['manifests']['train'], 'panels': panels,
              'panel_image_overlap': len({s['image'] for s in plans['2']} & {s['image'] for s in plans['4']}),
              'scope': 'Annotations and deterministic draws only; no pixels or model inference', 'epochs': {}}
    for epoch in range(1, cfg['epochs']):
        counts = Counter()
        for row in rows:
            choice = augmentation.decision(row, cfg, epoch, 'fixed_local', None)
            counts['images'] += 1
            counts['positive_images'] += bool(row['targets'])
            counts['opportunities'] += choice['opportunity']
            candidate = geometry.chosen_candidate(row, cfg['geometry'], cfg['seed'], epoch)
            try:
                annotations = geometry.prepared_annotations(row, 640)
            except geometry.InvalidAnnotation:
                counts['invalid_annotation_fallback'] += 1
                continue
            verdict = geometry.evaluate(annotations, candidate, 640)
            counts[verdict['reason']] += 1
            boxes = verdict['boxes'] if verdict['accepted'] else np.asarray([a['box_xyxy'] for a in annotations]).reshape(-1, 4)
            boxes = boxes.astype(np.float32)
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all()
            assert ((boxes >= 0) & (boxes <= 640)).all()
            counts['target_boxes_preserved'] += len(row['targets'])
        result['epochs'][str(epoch+1)] = dict(counts)
        print(f'METADATA epoch={epoch+1}: {dict(counts)}', flush=True)
    Path(__file__).with_name('metadata_preflight.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__': main()
