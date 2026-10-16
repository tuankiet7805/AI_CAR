"""Khung script hiển thị ảnh và bounding box để kiểm tra nhãn.

TODO: Bổ sung logic đọc ảnh và nhãn YOLO.
"""
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/yolo")
# REVIEW [P2]: ROOT/OUT phụ thuộc cwd như converter. Từ repo root sẽ đọc sai
# thư mục data/yolo; nên resolve cả hai path theo vị trí file trong thư mục ai/.
OUT = Path("reports/overlay")
OUT.mkdir(parents=True, exist_ok=True)

images = list((ROOT / "images" / "train").glob("*"))
# REVIEW [P2]: random.sample(..., 20) gây ValueError nếu có dưới 20 ảnh,
# bao gồm trường hợp ROOT sai hoặc chưa convert. Nên báo rõ khi không có ảnh,
# dùng min(20, len(images)) khi có ít ảnh; thêm seed để review lại cùng mẫu.
for p in random.sample(images, 20):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    lbl = ROOT / "labels" / "train" / f"{p.stem}.txt"
    for line in lbl.read_text().splitlines():
        cls, *xy = line.split()
        pts = (np.array(xy, dtype=float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, cls, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / p.name), img)
