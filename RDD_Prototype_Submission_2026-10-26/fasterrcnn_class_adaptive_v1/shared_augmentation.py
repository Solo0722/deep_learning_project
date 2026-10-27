"""Fixed class-shared appearance edits with per-image/epoch deterministic draws."""
import copy
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageFilter

FAMILIES = ("photometric", "blur_noise")
OPERATIONS = (
    "brightness_down", "brightness_up", "contrast_down", "contrast_up",
    "gamma_brighten", "gamma_darken", "gaussian_blur", "gaussian_noise",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_policy(policy):
    if not 0 <= policy["probability"] <= 1 or not math.isfinite(policy["probability"]):
        raise ValueError("Invalid augmentation probability")
    if type(policy["warmup_epochs"]) is not int or policy["warmup_epochs"] < 0:
        raise ValueError("Invalid augmentation warmup")
    if set(policy["operations"]) != set(OPERATIONS):
        raise ValueError("Unexpected operation bank")
    for name, parameters in policy["operations"].items():
        expected_family = "blur_noise" if name.startswith("gaussian_") else "photometric"
        key = "sigma_pixels" if name == "gaussian_blur" else "sigma_unit" if name == "gaussian_noise" else "gamma" if name.startswith("gamma_") else "factor"
        if set(parameters) != {"family", key} or parameters["family"] != expected_family:
            raise ValueError("Invalid operation parameters/family")
        if not math.isfinite(parameters[key]) or parameters[key] <= 0:
            raise ValueError("Invalid operation strength")


def random_word(key, purpose, counter=0):
    return int.from_bytes(hashlib.sha256(key + encoded([purpose, counter])).digest()[:8], "big")


def choice_index(key, purpose, count):
    # Rejection avoids a modulo bias when the operation count is not a power of two.
    limit = 2**64 - (2**64 % count)
    counter = 0
    while True:
        word = random_word(key, purpose, counter)
        if word < limit:
            return word % count
        counter += 1


def decision(row, policy, seed, epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid zero-based epoch")
    result = {"image": row["image"], "epoch": epoch+1, "target_bearing": bool(row["targets"]),
              "operation": "identity", "family": "identity", "parameters": {}, "noise_seed": None}
    if row["split"] != "train":
        result["reason"] = "validation"
    elif policy is None:
        result["reason"] = "disabled"
    elif not row["targets"]:
        result["reason"] = "no_target"
    elif epoch < policy["warmup_epochs"]:
        result["reason"] = "warmup"
    else:
        key = encoded([policy["version"], seed, epoch, row["image"]])
        if random_word(key, "application") >= int(policy["probability"] * 2**64):
            result["reason"] = "probability"
        else:
            family = FAMILIES[choice_index(key, "family", len(FAMILIES))]
            names = [name for name in OPERATIONS if policy["operations"][name]["family"] == family]
            name = names[choice_index(key, "operation", len(names))]
            result.update(reason="applied", operation=name, family=family,
                          parameters={k: v for k, v in policy["operations"][name].items() if k != "family"},
                          noise_seed=random_word(key, "noise") if name == "gaussian_noise" else None)
    return result


def transform_content(rgb, operation, parameters, noise_seed=None):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HWC RGB uint8 content")
    if operation == "identity":
        return rgb.copy()
    if operation not in OPERATIONS:
        raise ValueError("Unknown augmentation operation")
    if operation == "gaussian_blur":
        with Image.fromarray(rgb) as image:
            with image.filter(ImageFilter.GaussianBlur(radius=parameters["sigma_pixels"])) as blurred:
                return np.array(blurred, dtype=np.uint8, copy=True)
    values = rgb.astype(np.float64) / 255.
    if operation.startswith("brightness_"):
        output = values * parameters["factor"]
    elif operation.startswith("contrast_"):
        center = np.mean(.299*values[:, :, 0] + .587*values[:, :, 1] + .114*values[:, :, 2])
        output = center + parameters["factor"] * (values-center)
    elif operation.startswith("gamma_"):
        output = np.power(values, parameters["gamma"])
    else:
        if type(noise_seed) is not int or noise_seed < 0:
            raise ValueError("Noise requires an explicit nonnegative seed")
        generator = np.random.Generator(np.random.PCG64(noise_seed))
        output = values + generator.normal(0., parameters["sigma_unit"], size=values.shape)
    return np.rint(np.clip(output, 0., 1.)*255.).astype(np.uint8)


def apply_example(example, selected):
    if selected["image"] != example.record["image"]:
        raise ValueError("Decision belongs to a different image")
    if selected["operation"] == "identity":
        result = replace(example, rgb=example.rgb.copy())
    else:
        if example.record["split"] != "train" or not len(example.boxes):
            raise ValueError("Only target-bearing training images can be augmented")
        geometry = example.geometry
        if geometry is None:
            raise ValueError("Prepare image geometry before augmentation")
        left, top = geometry["pad_left"], geometry["pad_top"]
        width, height = geometry["resized_width"], geometry["resized_height"]
        content = example.rgb[top:top+height, left:left+width]
        canvas = example.rgb.copy()
        canvas[top:top+height, left:left+width] = transform_content(
            content, selected["operation"], selected["parameters"], selected["noise_seed"])
        result = replace(example, rgb=canvas)
    result.augmentation_decision = copy.deepcopy(selected)
    result.augmentation_changed = not np.array_equal(example.rgb, result.rgb)
    return result


def empty_trace():
    return {"counts": {}, "decision_sha256": "0"*64, "changed_images": 0}


def record_decision(trace, selected, changed=False):
    keys = ["images", "operation/"+selected["operation"], "reason/"+selected["reason"]]
    if selected["target_bearing"]:
        keys.append("target_images")
    if selected["operation"] != "identity":
        keys.extend(("applied_images", "family/"+selected["family"]))
    for key in keys:
        trace["counts"][key] = trace["counts"].get(key, 0) + 1
    trace["decision_sha256"] = hashlib.sha256(bytes.fromhex(trace["decision_sha256"]) + encoded(selected)).hexdigest()
    trace["changed_images"] += int(changed)


def verify_pairing(configuration, reference):
    allowed = {"augmentation", "augmentation_policy", "version", "purpose", "recipe_status", "comparator", "sources"}
    if {k: v for k, v in configuration.items() if k not in allowed} != {k: v for k, v in reference.items() if k not in allowed}:
        raise ValueError("Non-augmentation settings differ from the identity comparator")
    if configuration["augmentation"] != "fixed_shared_image_level":
        raise ValueError("Unexpected augmentation condition")
    validate_policy(configuration["augmentation_policy"])
