"""Unmodified eval predictions plus auxiliary positive-ROI object losses."""
from contextlib import contextmanager
import hashlib
import math

import torch
from torch.nn import functional as F
from torchvision.ops import box_iou

import checkpoints
from integration import require
from rdd_data import torchvision_collate


def tensor_digest(entries):
    result = hashlib.sha256()
    for name, value in entries:
        result.update(name.encode())
        if value is None:
            result.update(b'none')
        else:
            value = value.detach().cpu().contiguous()
            result.update(str((value.dtype, tuple(value.shape))).encode())
            result.update(value.numpy().tobytes())
    return result.hexdigest()


@contextmanager
def frozen_probe(model):
    modes = [(m, m.training) for m in model.modules()]
    rng = checkpoints.rng_state()
    state = tensor_digest(model.state_dict().items())
    grads = tensor_digest((n, p.grad) for n, p in model.named_parameters())
    model.eval()
    try:
        with torch.inference_mode():
            yield {'model_state_sha256': state, 'gradients_sha256': grads}
    finally:
        for module, training in modes:
            module.training = training
        try:
            require(tensor_digest(model.state_dict().items()) == state, 'Probe changed model parameters/buffers')
            require(tensor_digest((n, p.grad) for n, p in model.named_parameters()) == grads, 'Probe changed gradients')
            checkpoints.assert_same(checkpoints.rng_state(), rng, 'probe RNG')
        finally:
            checkpoints.restore_rng(rng)


def match_detections(prediction, boxes, labels, confidence, iou_threshold):
    boxes, labels = boxes.detach().cpu(), labels.detach().cpu()
    pred = {k: v.detach().cpu() for k, v in prediction.items()}
    n = len(pred['scores'])
    require(pred['boxes'].shape == (n, 4) and pred['labels'].shape == (n,), 'Malformed probe predictions')
    require(torch.isfinite(pred['scores']).all().item() and torch.isfinite(pred['boxes']).all().item(), 'Nonfinite probe predictions')
    require(((pred['scores'] >= 0) & (pred['scores'] <= 1)).all().item(), 'Invalid probe scores')
    require(((pred['labels'] >= 1) & (pred['labels'] <= 4)).all().item(), 'Invalid probe classes')
    require((pred['boxes'][:, 2:] > pred['boxes'][:, :2]).all().item(), 'Invalid predicted box geometry')
    order = sorted((i for i in range(n) if float(pred['scores'][i]) >= confidence), key=lambda i: (-float(pred['scores'][i]), i))
    overlaps = box_iou(pred['boxes'], boxes)
    detected, ious, fp = [False]*len(boxes), [0.]*len(boxes), 0
    for i in order:
        candidates = [j for j in range(len(boxes)) if not detected[j] and labels[j] == pred['labels'][i] and float(overlaps[i, j]) >= iou_threshold]
        if not candidates:
            fp += 1
            continue
        j = max(candidates, key=lambda j: (float(overlaps[i, j]), -j))
        detected[j], ious[j] = True, float(overlaps[i, j])
    return {'detected': detected, 'iou': ious, 'false_positives': fp, 'confident_predictions': len(order)}


def positive_rois(roi_heads, proposals, targets):
    boxes = [t['boxes'] for t in targets]
    labels = [t['labels'] for t in targets]
    rois, assignments = [], []
    for proposal, gt, gt_labels in zip(proposals, boxes, labels):
        require(len(gt) > 0, 'Probe panel image has no annotated targets')
        if len(proposal):
            matched, assigned_labels = roi_heads.assign_targets_to_proposals([proposal], [gt], [gt_labels])
            matches, lab = matched[0], assigned_labels[0]
        else:
            # An empty RPN output is a missed-object case, not an absent loss target.
            matches = torch.empty(0, dtype=torch.int64, device=gt.device)
            lab = torch.empty(0, dtype=torch.int64, device=gt.device)
        keep = lab > 0
        # A GT anchor per object is explicitly self-assigned, including identical GT boxes.
        rois.append(torch.cat((proposal[keep], gt), dim=0))
        assignments.append(torch.cat((matches[keep], torch.arange(len(gt), device=gt.device))))
    return rois, assignments


def object_loss_values(logits, regression, encoded_boxes, labels, assignments, object_counts):
    all_labels = torch.cat(labels)
    encoded = torch.cat(encoded_boxes)
    require(logits.shape == (len(all_labels), 5) and regression.shape == (len(all_labels), 20), 'Unexpected RDD ROI predictor')
    classification = F.cross_entropy(logits, all_labels, reduction='none')
    chosen = regression.reshape(-1, 5, 4)[torch.arange(len(all_labels), device=regression.device), all_labels]
    localization = F.smooth_l1_loss(chosen, encoded, beta=1/9, reduction='none').sum(1)
    values = (classification+localization).detach().cpu().tolist()
    require(all(math.isfinite(v) and v >= 0 for v in values), 'Nonfinite positive ROI loss')
    result, start = [], 0
    for assigned, count in zip(assignments, object_counts):
        bins = [[] for _ in range(count)]
        for index, loss in zip(assigned.detach().cpu().tolist(), values[start:start+len(assigned)]):
            bins[index].append(loss)
        require(all(bins), 'Annotated object missing supervised loss')
        result.append({'loss': [math.fsum(v)/len(v) for v in bins], 'roi_count': [len(v) for v in bins]})
        start += len(assigned)
    require(start == len(values), 'ROI loss partition mismatch')
    return result


def observe(model, examples, settings, device, verify_prediction_parity=False):
    require(not model.training and not torch.is_grad_enabled(), 'Probe must run in frozen eval/inference context')
    images, targets, _ = torchvision_collate(examples)
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    captured = []
    handle = model.roi_heads.register_forward_pre_hook(lambda module, args: captured.append(args[:3]))
    try:
        predictions = model(images)  # The ordinary inference path; GT never enters detection proposals.
    finally:
        handle.remove()
    require(len(captured) == 1 and len(predictions) == len(examples), 'Unexpected detector forward structure')
    features, proposals, shapes = captured[0]
    require([tuple(im.shape[-2:]) for im in images] == list(shapes), 'Probe altered prepared geometry')
    if verify_prediction_parity:
        ordinary = model(images)
        for a, b in zip(predictions, ordinary):
            for key in ('boxes', 'scores', 'labels'):
                require(torch.equal(a[key], b[key]), 'ROI observation hook changed standard predictions')
    rois, assignments = positive_rois(model.roi_heads, proposals, targets)
    pooled = model.roi_heads.box_roi_pool(features, rois, shapes)
    logits, regression = model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))
    labels = [t['labels'][a] for t, a in zip(targets, assignments)]
    encoded = model.roi_heads.box_coder.encode([t['boxes'][a] for t, a in zip(targets, assignments)], rois)
    losses = object_loss_values(logits, regression, encoded, labels, assignments, [len(t['boxes']) for t in targets])
    return [{**match_detections(pred, t['boxes'], t['labels'], settings['confidence'], settings['iou']), **loss}
            for pred, t, loss in zip(predictions, targets, losses)]
