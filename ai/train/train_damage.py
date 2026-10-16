"""Train CarDD segmentation; chọn best.pt theo validation, không dùng test để chọn model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import AI_ROOT, prepare_dataset, select_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=AI_ROOT / 'yolov8n-seg.pt')
    p.add_argument('--data', type=Path, default=AI_ROOT / 'configs/data.yaml')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='auto')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--name', default='cardd_30epochs')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.epochs <= 0 or a.imgsz <= 0 or a.batch <= 0:
        p.error('epochs, imgsz and batch must be positive')
    if Path(a.name).name != a.name or a.name in {'.', '..'}:
        p.error('name must be a single directory name')
    if not a.model.is_file():
        p.error(f'Checkpoint does not exist: {a.model}')
    from ultralytics import YOLO
    model = YOLO(str(a.model.resolve()))
    if model.task != 'segment':
        p.error('Checkpoint must be a segmentation model')
    run = AI_ROOT / 'runs/damage' / a.name
    if run.exists():
        p.error(f'Run already exists: {run}; choose a new --name')
    dataset = prepare_dataset(a.data, run / 'dataset.yaml')
    # exist_ok chỉ cho folder mới vừa tạo ở trên; không ghi đè thí nghiệm cũ.
    model.train(data=str(dataset), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=select_device(a.device), workers=a.workers, cache=False,
                patience=a.patience, project=str(run.parent), name=a.name,
                exist_ok=True, seed=a.seed, plots=True)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training ended without best.pt')
    print(f'Validation-selected checkpoint: {best}')


if __name__ == '__main__':
    main()
