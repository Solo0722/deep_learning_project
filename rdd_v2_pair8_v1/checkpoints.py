"""Exact state restoration into a newly constructed model; no pickle objects in our checkpoints."""
import hashlib
import random

import numpy as np
import torch

from integration import require


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def assert_same(actual, expected, path="state"):
    if torch.is_tensor(expected):
        require(torch.is_tensor(actual) and actual.dtype == expected.dtype
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()), f"Changed {path}")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Changed keys: {path}")
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), f"Changed length/type: {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    else:
        require(actual == expected, f"Changed {path}")


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    require(len(state["cuda"]) == (torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "Checkpoint CUDA RNG device count differs")
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, model, optimizer, model_name, step, context):
    payload = {"format_version": 1, "model_name": model_name, "completed_steps": step,
               "context": context, "model_training": model.training,
               "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
               "rng": rng_state()}
    # An interrupted write has no successful metadata record and is not resumable.
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return {"file": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size,
            "completed_steps": step, "model_name": model_name}


def restore(path, expected, model, optimizer, model_name, context):
    require(file_hash(path) == expected["sha256"], "Checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and payload["model_name"] == model_name,
            "Checkpoint format or model mismatch")
    require(payload["context"] == context, "Checkpoint input/protocol context mismatch")
    require(payload["completed_steps"] == expected["completed_steps"], "Checkpoint step mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    model.train(payload["model_training"])
    assert_same(model.state_dict(), payload["model"], "model")
    assert_same(optimizer.state_dict(), payload["optimizer"], "optimizer")
    restore_rng(payload["rng"])
    assert_same(rng_state(), payload["rng"], "rng")
    return {"model_state_exact": True, "optimizer_state_exact": True, "rng_state_exact": True,
            "next_step": payload["completed_steps"] + 1}
