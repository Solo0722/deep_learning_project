"""Exact state fingerprints and bounded numeric evidence for the diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def leaves(value, path='$'):
    if isinstance(value, dict):
        yield path + '/@container', 'dict'
        for k in sorted(value, key=lambda x: (type(x).__name__, repr(x))):
            yield from leaves(value[k], path + '/' + type(k).__name__ + ':' + repr(k))
    elif isinstance(value, (tuple, list)):
        yield path + '/@container', type(value).__name__
        for i, v in enumerate(value):
            yield from leaves(v, path + '/' + str(i))
    else:
        yield path, value


def raw_tensor(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def leaf_bytes(value):
    if torch.is_tensor(value):
        header = json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        return header + b'\0' + raw_tensor(value)
    return (type(value).__name__ + ':' + repr(value)).encode()


def digest(value):
    result = hashlib.sha256()
    for path, v in leaves(value):
        key, data = path.encode(), leaf_bytes(v)
        result.update(len(key).to_bytes(8, 'little')); result.update(key)
        result.update(len(data).to_bytes(8, 'little')); result.update(data)
    return result.hexdigest()


def finite(value):
    for path, v in leaves(value):
        if torch.is_tensor(v) and not torch.isfinite(v).all().item():
            raise ValueError('Nonfinite tensor at ' + path)
        if isinstance(v, float) and not np.isfinite(v):
            raise ValueError('Nonfinite scalar at ' + path)


def overview(value):
    result = {'sha256': digest(value), 'tensor_count': 0, 'tensor_elements': 0}
    for _, v in leaves(value):
        if torch.is_tensor(v):
            result['tensor_count'] += 1
            result['tensor_elements'] += v.numel()
    return result


def compare(expected, actual, samples=8):
    """Returns a JSON report and the first differing tensor pair, if available.

    Equality is byte-exact; magnitudes are descriptive, with no pass tolerance.
    Shape/dtype/key mismatches are reported rather than broadcast or truncated.
    """
    left, right = dict(leaves(expected)), dict(leaves(actual))
    eh, ah = digest(expected), digest(actual)
    result = {'equal': eh == ah, 'expected_sha256': eh, 'actual_sha256': ah,
              'missing_paths': sorted(left.keys()-right.keys()),
              'unexpected_paths': sorted(right.keys()-left.keys()), 'changes': []}
    first_pair = None
    if result['equal']:
        return result, first_pair
    for path in sorted(left.keys() & right.keys()):
        a, b = left[path], right[path]
        if leaf_bytes(a) == leaf_bytes(b):
            continue
        entry = {'path': path}
        if torch.is_tensor(a) and torch.is_tensor(b):
            entry.update(expected_shape=list(a.shape), actual_shape=list(b.shape),
                         expected_dtype=str(a.dtype), actual_dtype=str(b.dtype))
            if a.shape != b.shape or a.dtype != b.dtype:
                entry['kind'] = 'tensor_structure'
            else:
                finite(a); finite(b)
                da, db = a.detach().cpu().reshape(-1).double(), b.detach().cpu().reshape(-1).double()
                delta = db-da
                n = delta.numel()
                norm = float(torch.linalg.vector_norm(da))
                diff_norm = float(torch.linalg.vector_norm(delta))
                positions = torch.topk(delta.abs(), min(samples, n), sorted=True).indices.tolist() if n else []
                entry.update(kind='tensor_values', elements=n,
                    numerically_changed_elements=int(torch.count_nonzero(delta)),
                    max_abs=float(delta.abs().max()) if n else 0.,
                    l2=diff_norm, reference_l2=norm,
                    relative_l2=diff_norm/norm if norm else None,
                    values=[{'flat_index': i, 'expected': float(da[i]), 'actual': float(db[i])} for i in positions])
                if first_pair is None:
                    first_pair = (path, a, b)
        else:
            entry.update(kind='value_or_type', expected=repr(a), actual=repr(b))
        result['changes'].append(entry)
    result['changed_paths'] = len(result['changes']) + len(result['missing_paths']) + len(result['unexpected_paths'])
    return result, first_pair


def export_pair(directory, name, pair, limit):
    if pair is None:
        return None
    path, a, b = pair
    size = a.numel()*a.element_size() + b.numel()*b.element_size()
    if size > limit:
        return {'path': path, 'bytes_uncompressed': size, 'saved': False,
                'reason': 'Full tensor pair exceeds declared cap; numeric samples remain in comparison JSON.'}
    target = Path(directory) / (name + '.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        np.savez_compressed(handle, expected=a.detach().cpu().numpy(), actual=b.detach().cpu().numpy())
    return {'path': path, 'file': target.name, 'bytes_uncompressed': size,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'saved': True}
