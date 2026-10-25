# AutoClaimAI — workspace AI

**Hướng dẫn từng bước:** [Cài môi trường, kiểm tra dữ liệu, train, đánh giá và dự đoán](docs/HUONG_DAN_CHAY.md).

## Cấu trúc thư mục sau khi chuẩn bị dữ liệu và chạy pipeline

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

## Data / Dataset

Dự án dùng **CarDD** cho bài toán segmentation với 6 lớp: `dent`, `scratch`,
`crack`, `glass_shatter`, `lamp_broken`, `tire_flat`.

**Dataset không đi kèm repo Git.** Thư mục `ai/data/` được loại khỏi Git bằng
`.gitignore`, nên sau khi clone cần tải và chuẩn bị dữ liệu riêng. Checkpoint
(`*.pt`), kết quả train, báo cáo và `_archive/` cũng không đi kèm repo;
cây thư mục phía trên mô tả môi trường đã chuẩn bị đầy đủ.

1. Truy cập [trang CarDD chính thức](https://cardd-ustc.github.io/), làm theo
   hướng dẫn tải và điều kiện sử dụng tại đó. Chọn dữ liệu **COCO** có ảnh và
   annotation polygon để dùng với converter của dự án.
2. Giải nén và sắp xếp dữ liệu theo cấu trúc bên dưới. Converter hiện đọc tên
   split `train`, `val`, `test` và các file JSON tương ứng. Nếu bản tải về dùng
   tên `train2017`, `val2017`, `test2017` hoặc `instances_*2017.json`, đổi tên
   thư mục và file tương ứng; giữ nguyên nội dung annotation và tên ảnh.

```text
ai/data/raw/CarDD/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── train/  # 2816 ảnh
├── val/    # 810 ảnh
└── test/   # 374 ảnh
```

3. Từ thư mục gốc repo, sau khi cài dependency theo
   [hướng dẫn chạy](docs/HUONG_DAN_CHAY.md), chuyển nhãn COCO sang YOLO segmentation:

```bash
python3 ai/scripts/coco_to_yolo.py
```

Lệnh này tạo `ai/data/yolo/images/{train,val,test}/` và
`ai/data/yolo/labels/{train,val,test}/`, đồng thời ghi lại `ai/configs/data.yaml`
với đường dẫn trên máy hiện tại. Không dùng nguyên đường dẫn tuyệt đối của máy
cũ trong YAML. Kiểm tra log mỗi split có `thiếu file 0` và `bỏ qua 0 polygon`;
nếu khác, kiểm tra dữ liệu trước khi train.

4. Kiểm tra thống kê và nhãn trực quan:

```bash
cd ai
python3 scripts/stats_cardd.py
python3 scripts/visualize_labels.py
```

Ảnh kiểm tra được lưu ở `ai/reports/overlay/`. Giữ nguyên split chính thức
2816/810/374; không chia lại dữ liệu. Nếu đã có dữ liệu YOLO đầy đủ, có thể bỏ qua
convert và sửa `path` trong `ai/configs/data.yaml` thành `data/yolo`
(đường dẫn tương đối được train/evaluate tính từ `ai/`).

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
