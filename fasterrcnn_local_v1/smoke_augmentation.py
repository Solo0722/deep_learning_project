"""Train-only previews and outside-union invariants for fixed local operations."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

import local_augmentation as aug


def forced_choice(example, cfg, operation, group_index):
    parameters = cfg["augmentation_policy"]["operations"][operation]
    selected = {"image": example.record["image"], "epoch": 1, "target_bearing": True,
                "operation": operation, "family": parameters["family"], "reason": "forced_preview",
                "parameters": {k: v for k, v in parameters.items() if k != "family"},
                "noise_seed": 42 if operation == "gaussian_noise" else None}
    return aug.attach_local(example.record, selected, cfg["seed"], 0, cfg["image_size"], group_index)


def check_result(example, selected):
    reference = copy.deepcopy(example)
    result = aug.apply_example(example, selected)
    mask = np.zeros(example.rgb.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in selected["local"]["rectangles"]:
        mask[y1:y2, x1:x2] = True
    if not (np.array_equal(example.rgb, reference.rgb)
            and np.array_equal(result.rgb[~mask], reference.rgb[~mask])
            and np.array_equal(result.boxes, reference.boxes)
            and np.array_equal(result.labels, reference.labels)
            and result.record == reference.record and result.geometry == reference.geometry):
        raise ValueError("Local edit changed reference, outside-union pixels, padding or targets")
    if result.rgb.dtype != np.uint8 or result.rgb.shape != reference.rgb.shape:
        raise ValueError("Local operation changed pixel dtype/shape")
    return result


def grid(path, previews):
    columns = 3
    canvas = Image.new("RGB", (columns*256, ((len(previews)+columns-1)//columns)*280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, pixels) in enumerate(previews):
        x, y = (index % columns)*256, (index // columns)*280
        draw.text((x+6, y+5), name, fill="black")
        with Image.fromarray(pixels) as im:
            with im.resize((256, 256), Image.Resampling.BILINEAR) as thumb:
                canvas.paste(thumb, (x, y+24))
    with path.open("xb") as handle:
        canvas.save(handle, format="PNG")
    canvas.close()


def check_operations(dataset, cfg, out):
    # Choose by geometry only, with deterministic ties; never by detector performance.
    candidates = [(i, aug.groups(row, cfg["image_size"])) for i, (_, row) in enumerate(dataset.rows) if row["targets"]]
    index, options = max(candidates, key=lambda item: max(len(g[0]) for g in item[1]))
    selected_index = max(range(len(options)), key=lambda i: len(options[i][0]))
    example = dataset[index]
    if example.augmentation_decision["operation"] != "identity":
        raise ValueError("Preview must start from unaugmented reference data")
    entries, previews = [], [("reference", example.rgb)]
    for operation in aug.OPERATIONS:
        selected = forced_choice(example, cfg, operation, selected_index)
        result = check_result(example, selected)
        entries.append({"operation": operation, "parameters": selected["parameters"],
                        "noise_seed": selected["noise_seed"], "local": selected["local"],
                        **result.augmentation_pixel_stats,
                        "rgb_sha256": hashlib.sha256(result.rgb.tobytes()).hexdigest()})
        previews.append((operation, result.rgb))
    grid(out / "augmentation_preview.png", previews)
    # Also exercise a disjoint group and the smallest available group (thin boxes remain eligible).
    separated = max(candidates, key=lambda item: len(item[1]))
    smallest = min(((i, j, g[2]) for i, gs in candidates for j, g in enumerate(gs)), key=lambda x: x[2])
    cases = [("overlap", index, selected_index), ("separated", separated[0], 0), ("smallest", smallest[0], smallest[1])]
    geometry_entries, geometry_previews = [], []
    for label, index, group_index in cases:
        reference = dataset[index]
        selected = forced_choice(reference, cfg, "brightness_down", group_index)
        result = check_result(reference, selected)
        mask = np.zeros_like(reference.rgb)
        (x1, y1, x2, y2), units = aug.union_weights(selected["local"]["rectangles"])
        mask[y1:y2, x1:x2] = np.rint(units[..., None]*127.5).astype(np.uint8)
        geometry_previews.extend([(label+" reference", reference.rgb), (label+" alpha", mask), (label+" edited", result.rgb)])
        geometry_entries.append({"case": label, "image": reference.record["image"],
                                 "local": selected["local"], **result.augmentation_pixel_stats})
    grid(out / "local_geometry_preview.png", geometry_previews)
    report = {"status": "passed", "image": example.record["image"], "operations": entries,
              "geometry_cases": geometry_entries, "outside_union_unchanged": True,
              "targets_and_padding_unchanged": True,
              "scope": "Forced training-image checks; not sampled exposure, semantic safety or efficacy evidence"}
    with (out / "augmentation_operation_check.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("All 8 local operations: targets/padding/outside-union checks PASS", flush=True)
    return report
