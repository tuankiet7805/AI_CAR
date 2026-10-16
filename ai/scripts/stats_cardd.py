import json
from collections import Counter
from pathlib import Path

ANN_DIR = Path("data/raw/CarDD/annotations")

for split in ("train", "val", "test"):
    path = ANN_DIR / f"instances_{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    per_class = Counter(cats[a["category_id"]] for a in data["annotations"])

    # tỉ lệ diện tích hư hỏng / ảnh -> cho biết lớp nào "nhỏ, khó"
    ratios = {}
    for a in data["annotations"]:
        img = images[a["image_id"]]
        ratios.setdefault(cats[a["category_id"]], []).append(
            a["area"] / (img["width"] * img["height"])
        )

    print(f"\n== {split}: {len(images)} ảnh, {len(data['annotations'])} instance ==")
    for name, n in per_class.most_common():
        r = sorted(ratios[name])
        print(f"{name:15s} {n:6d}  median area {r[len(r)//2]:.4f}")