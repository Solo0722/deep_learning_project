"""Fetch pinned public validation inputs into workspace tmp; do not install packages globally."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[2] / "tmp/detection_learning_v1_weights"
root.mkdir(parents=True, exist_ok=True)
items = [
    ("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
     "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
     "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384"),
    ("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
     "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
]
with urllib.request.urlopen("https://pypi.org/pypi/ultralytics-opencv-headless/8.4.182/json", timeout=30) as response:
    metadata = json.load(response)
wheel = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
items.append((wheel["filename"], wheel["url"], wheel["digests"]["sha256"]))
for name, url, expected in items:
    path = root / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as out:
            while block := response.read(1024*1024):
                out.write(block)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch for {name}; preserve and inspect the partial file")
    print(name, path.stat().st_size, actual, flush=True)
target = root / "ultralytics_8_4_182"
target.mkdir(exist_ok=False)
with zipfile.ZipFile(root / wheel["filename"]) as archive:
    for entry in archive.infolist():
        if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
            raise ValueError("Unsafe wheel member")
    archive.extractall(target)
print("Isolated import directory:", target, flush=True)
