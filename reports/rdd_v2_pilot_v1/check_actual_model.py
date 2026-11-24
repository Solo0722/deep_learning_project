"""Local CPU computational check; synthetic inputs/random weights, not efficacy."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'rdd_v2_pilot_v1'))
sys.path.insert(0, str(ROOT/'rdd_v2_pilot_v1/tests'))
import numpy as np
import torch
import torchvision
import augmentation
import baseline as b
import checkpoints
import diagnostic
import observer
import pilot
import rdd_data
import test_baseline
from test_pilot import policy


def main():
    torch.set_num_threads(2)
    fixture = test_baseline.BaselineTests(); fixture.setUp()
    try:
        cfg = json.loads((pilot.PACKAGE/'protocol.json').read_text())
        cfg['application_probability'] = 1.
        examples = [rdd_data.prepare_example(rdd_data.load_example(fixture.root, row, i), 640)
                    for i, row in enumerate([fixture.train_rows[0], fixture.train_rows[-1]])]
        model, _, effective = b.build_model(b.MODEL, 640, 'cpu', cfg['evaluation'])
        optimizer = torch.optim.SGD(model.parameters(), lr=.0025, momentum=.9, weight_decay=.0005)
        payload = {'format': 1, 'model': checkpoints.cpu_tree(model.state_dict()),
                   'optimizer': checkpoints.cpu_tree(optimizer.state_dict()), 'rng': checkpoints.rng_state()}
        results, hashes = {}, []
        for arm in cfg['arms']:
            start = time.monotonic()
            pilot.fork_state(model, optimizer, payload)
            transformed = [augmentation.apply(ex, cfg, 1, arm, policy()) for ex in examples]
            trace = augmentation.empty_trace()
            for ex in transformed: augmentation.record_trace(trace, ex.pilot_trace)
            hashes.append(trace['pairing_sha256'])
            images, targets, _ = b.collate(transformed)
            head = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
            losses, norm = b.update(model, optimizer, images, targets, 'cpu')
            assert not torch.equal(head, model.roi_heads.box_predictor.cls_score.weight)
            with observer.frozen(model, optimizer) as audit:
                detector = observer.CurrentDetector(model, 'cpu', 2)
                predictions = detector.predict([ex.rgb for ex in transformed])
                for pred, ex in zip(predictions, transformed):
                    diagnostic.match_prediction(pred, ex.boxes, ex.labels, cfg['controller'])
            results[arm] = {'losses': losses, 'gradient_norm': norm, 'classifier_changed': True,
                            'predictions': [len(p['scores']) for p in predictions], 'observer': audit,
                            'target_counts': [len(ex.boxes) for ex in transformed],
                            'elapsed_seconds': time.monotonic()-start}
            print(f'ACTUAL CPU MODEL {arm}: PASS ({time.monotonic()-start:.1f}s)', flush=True)
        assert len(set(hashes)) == 1
        result = {'status': 'pass', 'scope': 'Random-weight actual Faster R-CNN, synthetic RGB, 640 pixels, positive plus negative batch, one optimizer step per arm; not learning or pinned CUDA validation',
                  'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__},
                  'paired_inputs': True, 'arms': results, 'effective_model': effective}
        Path(__file__).with_name('actual_model_check.json').write_text(json.dumps(result, indent=2)+'\n')
    finally: fixture.temp.cleanup()


if __name__ == '__main__': main()
