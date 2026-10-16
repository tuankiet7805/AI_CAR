# AutoClaimAI — workspace AI

**Hướng dẫn từng bước:** [Cài môi trường, kiểm tra dữ liệu, train, đánh giá và dự đoán](docs/HUONG_DAN_CHAY.md).

## Thư mục đang dùng

```text
ai/
├── configs/       # Bảng lớp và cấu hình dataset
├── data/
│   ├── raw/CarDD/ # Dataset COCO gốc, giữ split chính thức
│   └── yolo/      # Ảnh và nhãn polygon đã chuyển đổi
├── scripts/       # Thống kê, chuyển nhãn và vẽ ảnh kiểm tra
├── train/         # train_damage.py
├── inference/     # CLI/API nhận ảnh, mapping và xuất JSON
├── reports/       # Overlay, thư mục metrics và phân tích lỗi
├── runs/          # Kết quả và checkpoint chạy thử
├── yolov8n-seg.pt # Trọng số nền hiện có
└── requirements.txt
docs/             # SRS, proposal và ảnh tham khảo
_archive/         # Repo tham khảo, dữ liệu SOD và notebook thử nghiệm
```

## Chạy script

Script thống kê và vẽ nhãn hiện dùng đường dẫn tương đối, nên chạy từ thư mục `ai/`:

```bash
cd ai
python3 scripts/stats_cardd.py
python3 scripts/visualize_labels.py
```

`scripts/coco_to_yolo.py` tạo lại nhãn và ghi lại `configs/data.yaml`.
`train/train_damage.py` chạy huấn luyện segmentation. Dependency được khai báo trong
`ai/requirements.txt`. Train, evaluate và inference đã resolve đường dẫn từ vị trí code;
script thống kê và overlay vẫn cần chạy từ `ai/`. CLI và API nhận ảnh được mô tả
trong hướng dẫn chi tiết.

Giữ cả dữ liệu gốc lẫn dữ liệu YOLO để kiểm tra và huấn luyện. Không chia lại
split CarDD (2816/810/374). Chi tiết lớp và mapping xem `ai/configs/classes.md`.

## Những mục đã dọn

- Xóa file khung `convert_coco_to_yolo.py` vì đã có `coco_to_yolo.py` thực thi.
- Xóa file khung `split.py`, `train_parts.py` và thư mục trống `data/processed/`
  vì pipeline hiện tại chưa sử dụng.
- Chuyển tài liệu và ảnh tham khảo vào `docs/`.
- Chuyển repo CarDD tham khảo, bộ SOD và notebook EDNet vào `_archive/`, giữ nguyên nội dung.
- Giữ kết quả train, checkpoint và ảnh overlay để review.

Việc gom vào `_archive/` giúp gọn thư mục làm việc, không giải phóng dung lượng dataset.
