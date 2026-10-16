"""Đánh giá val/test riêng; xuất metric tổng, AP từng lớp và hash checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import AI_ROOT, CLASS_NAMES, prepare_dataset, select_device, validate_model


def metric_rows(metric):
    # Không dùng maps cho lớp vắng: thư viện có thể điền mAP tổng vào lớp vắng.
    return {int(cls): [float(x) for x in metric.class_result(i)]
            for i, cls in enumerate(metric.ap_class_index)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--data", type=Path, default=AI_ROOT / "configs/data.yaml")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if (a.output / "metrics.json").exists():
        p.error("Report exists; use a new output directory")
    if not a.model.is_file():
        p.error(f"Checkpoint does not exist: {a.model}")
    from ultralytics import YOLO
    model = YOLO(str(a.model.resolve()))
    validate_model(model)
    dataset = prepare_dataset(a.data, a.output / "dataset.yaml")
    metrics = model.val(data=str(dataset), split=a.split, imgsz=a.imgsz,
                        batch=a.batch, device=select_device(a.device), workers=0,
                        project=str(a.output.resolve()), name="validation", plots=True)
    box, mask = metric_rows(metrics.box), metric_rows(metrics.seg)
    report = {"checkpoint": str(a.model.resolve()),
              "checkpoint_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
              "split": a.split, "imgsz": a.imgsz, "box_mAP50": float(metrics.box.map50),
              "box_mAP50_95": float(metrics.box.map), "mask_mAP50": float(metrics.seg.map50),
              "mask_mAP50_95": float(metrics.seg.map), "per_class": []}
    for i, name in enumerate(CLASS_NAMES):
        row = {"class_id": i, "class_name": name}
        for prefix, values in (("box", box.get(i)), ("mask", mask.get(i))):
            for key, value in zip(("precision", "recall", "AP50", "AP50_95"), values or [None]*4):
                row[f"{prefix}_{key}"] = value
        report["per_class"].append(row)
    (a.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (a.output / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["per_class"][0]))
        writer.writeheader()
        writer.writerows(report["per_class"])
    print(json.dumps({k:v for k,v in report.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
