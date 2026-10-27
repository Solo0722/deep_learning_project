"""Save a train-only operation preview and verify unchanged spatial targets."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

import shared_augmentation as aug


def check_operations(dataset, cfg, out):
    index = next(i for i, (_, row) in enumerate(dataset.rows) if row["targets"])
    example = dataset[index]
    reference = example.rgb.copy()
    boxes, labels, record = example.boxes.copy(), example.labels.copy(), copy.deepcopy(example.record)
    geometry = example.geometry
    mask = np.zeros(example.rgb.shape[:2], dtype=bool)
    x, y = geometry["pad_left"], geometry["pad_top"]
    mask[y:y+geometry["resized_height"], x:x+geometry["resized_width"]] = True
    entries, previews = [], [("reference", reference)]
    for name in aug.OPERATIONS:
        parameters = cfg["augmentation_policy"]["operations"][name]
        selected = {"image": record["image"], "epoch": 1, "target_bearing": True,
                    "operation": name, "family": parameters["family"], "reason": "forced_preview",
                    "parameters": {k: v for k, v in parameters.items() if k != "family"}, "noise_seed": 42 if name == "gaussian_noise" else None}
        result = aug.apply_example(example, selected)
        if not (np.array_equal(example.rgb, reference) and np.array_equal(result.rgb[~mask], reference[~mask])
                and np.array_equal(result.boxes, boxes) and np.array_equal(result.labels, labels)
                and result.record == record and result.geometry == geometry):
            raise ValueError("Operation changed reference, padding or detection targets")
        if result.rgb.dtype != np.uint8 or result.rgb.shape != reference.shape:
            raise ValueError("Operation changed pixel dtype/shape")
        entries.append({"operation": name, "parameters": selected["parameters"], "noise_seed": selected["noise_seed"],
                        "changed_pixels": int(np.any(result.rgb != reference, axis=2).sum()),
                        "rgb_sha256": hashlib.sha256(result.rgb.tobytes()).hexdigest()})
        previews.append((name, result.rgb))
    canvas = Image.new("RGB", (3*256, 3*280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, pixels) in enumerate(previews):
        x, y = (index % 3)*256, (index // 3)*280
        draw.text((x+6, y+5), name, fill="black")
        with Image.fromarray(pixels) as im:
            with im.resize((256, 256), Image.Resampling.BILINEAR) as thumb:
                canvas.paste(thumb, (x, y+24))
    with (out / "augmentation_preview.png").open("xb") as handle:
        canvas.save(handle, format="PNG")
    canvas.close()
    report = {"status": "passed", "image": record["image"], "operations": entries,
              "targets_and_padding_unchanged": True,
              "scope": "Forced operation preview on one training image; not sampled training exposure or a semantic-safety guarantee"}
    with (out / "augmentation_operation_check.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("All 8 appearance operations: geometry/target/padding checks PASS", flush=True)
    return report
