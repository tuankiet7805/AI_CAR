"""Nhận ảnh/thư mục ảnh và xuất JSON hư hỏng; dùng checkpoint CarDD 6 lớp."""

import json
import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import canonical_name, validate_model, select_device
elif __package__.startswith("ai."):
    from ..runtime import canonical_name, validate_model, select_device
else:
    from runtime import canonical_name, validate_model, select_device

if __package__:
    from .schema import DamageType
else:
    from schema import DamageType


# REVIEW: Giữ riêng 6 lớp trong model. Việc gộp dưới đây chỉ xảy ra khi xuất JSON.
CARDD_TO_DAMAGE_TYPE = {
    "dent": DamageType.DENT,
    "scratch": DamageType.SCRATCH,
    "crack": DamageType.CRACK,
    "glass shatter": DamageType.CRACK,
    "lamp broken": DamageType.CRACK,
    "tire flat": None,  # Ngoài phạm vi output; vẫn giữ nhãn này khi train/evaluate.
}


def map_damage_type(class_name: str) -> Optional[DamageType]:
    """Trả None chỉ cho tire flat; tên không biết phải báo lỗi để tránh mất nhãn."""
    return CARDD_TO_DAMAGE_TYPE[canonical_name(class_name)]


def export_predictions(predictions: Iterable[Mapping[str, Any]]) -> str:
    """Xuất mảng JSON, thêm damage_type và giữ class_name gốc để truy vết.

    Ví dụ: [{"class_name": "glass shatter", "confidence": 0.9}]
    thành [{"class_name": "glass shatter", "confidence": 0.9,
            "damage_type": "CRACK"}]. Không sửa dict đầu vào.
    """
    output = []
    for prediction in predictions:
        damage_type = map_damage_type(prediction["class_name"])
        if damage_type is None:
            continue
        output.append({**prediction, "damage_type": damage_type.value})
    return json.dumps(output, ensure_ascii=False, allow_nan=False)


class DamagePredictor:
    """Nạp model một lần; không coi không phát hiện là xe không hỏng."""

    def __init__(self, model_path: str, device="auto", imgsz=640, confidence=0.25):
        from ultralytics import YOLO
        if not 0 <= confidence <= 1 or imgsz <= 0:
            raise ValueError("confidence must be in [0,1] and imgsz must be positive")
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model = YOLO(str(path))
        validate_model(self.model)
        self.model_path = str(path)
        self.device = select_device(device)
        self.imgsz, self.confidence = imgsz, confidence

    def predict(self, image_path: str) -> dict:
        import cv2
        path = Path(image_path).resolve()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.confidence,
                                    device=self.device, verbose=False)[0]
        predictions = []
        excluded = 0
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls = int(box.cls.item())
                name = canonical_name(result.names[cls])
                if map_damage_type(name) is None:
                    excluded += 1
                    continue
                polygon = result.masks.xy[i].tolist() if result.masks is not None else None
                predictions.append({"class_id": cls, "class_name": name,
                                    "confidence": float(box.conf.item()),
                                    "bbox_xyxy": box.xyxy[0].tolist(), "polygon_xy": polygon})
        return {"image": str(path), "width": image.shape[1], "height": image.shape[0],
                "detections": json.loads(export_predictions(predictions)),
                "excluded_tire_flat": excluded}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Checkpoint CarDD đã train")
    p.add_argument("--source", required=True, type=Path, help="File ảnh hoặc thư mục ảnh")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    sources = sorted(x for x in a.source.rglob("*") if x.suffix.lower() in
                     {".jpg", ".jpeg", ".png", ".webp"}) if a.source.is_dir() else [a.source]
    if not sources:
        p.error("No images found")
    predictor = DamagePredictor(str(a.model), a.device, a.imgsz, a.conf)
    output = {"schema_version": "1.0", "model": predictor.model_path,
              "confidence_threshold": a.conf, "imgsz": a.imgsz,
              "coordinate_system": "pixels; bbox_xyxy=[left,top,right,bottom]",
              "images": [predictor.predict(str(path)) for path in sources]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(f"Saved {len(sources)} image results to {a.output.resolve()}")


if __name__ == "__main__":
    main()
