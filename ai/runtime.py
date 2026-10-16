"""Đường dẫn và bảng lớp dùng chung; không phụ thuộc thư mục gọi lệnh."""
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat")


def canonical_name(name: str) -> str:
    # Hỗ trợ checkpoint cũ có '_' và annotation gốc có dấu cách.
    name = " ".join(name.strip().lower().replace("_", " ").split())
    if name not in CLASS_NAMES:
        raise ValueError(f"Unknown CarDD class: {name!r}")
    return name


def validate_model(model) -> None:
    if model.task != "segment":
        raise ValueError(f"Expected segmentation model, got {model.task!r}")
    actual = tuple(canonical_name(model.names[i]) for i in range(len(model.names)))
    if actual != CLASS_NAMES:
        raise ValueError(f"Checkpoint must contain the 6 CarDD classes in order, got {actual}")


def prepare_dataset(source: Path, destination: Path) -> Path:
    """Resolve YAML cho từng lượt chạy; giữ nguyên YAML nguồn và split chính thức."""
    import yaml
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    names = cfg["names"]
    if tuple(canonical_name(names[i]) for i in range(len(names))) != CLASS_NAMES:
        raise ValueError("Dataset class order does not match CarDD")
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = AI_ROOT / root
    for split, expected in (("train", 2816), ("val", 810), ("test", 374)):
        images = [p for p in (root / cfg[split]).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(images) != expected:
            raise ValueError(f"{split}: expected {expected} images, found {len(images)}")
        for image in images:
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
    cfg["path"] = str(root.resolve())
    # Giữ tên underscore như checkpoint cũ để cls_remap nạp đủ 6 lớp khi fine-tune.
    cfg["names"] = {i: name.replace(" ", "_") for i, name in enumerate(CLASS_NAMES)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination.resolve()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
