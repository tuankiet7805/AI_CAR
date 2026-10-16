# Hướng dẫn chạy AutoClaimAI

Code hiện hỗ trợ train segmentation, báo cáo AP từng lớp, chọn checkpoint bằng
validation, đánh giá test riêng, nhận ảnh qua CLI hoặc API và xuất JSON.
Các lệnh dưới dùng macOS/Linux (zsh/bash), chạy từ thư mục gốc project.

## 1. Chuẩn bị môi trường

```bash
cd /Users/nguyenson/Tuankiet
python3 -m venv ai/.venv
source ai/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ai/requirements.txt
python -m pip check
```

Nếu đã có môi trường đang chạy được project, có thể dùng lại. Đừng tạo lại môi
trường giữa một lượt train đang chạy. `requirements.txt` khai báo các package và
phiên bản quan sát được trên máy khi triển khai; không phải lock toàn bộ dependency
bắc cầu hoặc cam kết cùng bản PyTorch chạy trên mọi nền tảng.

Với RTX/NVIDIA, chọn bản PyTorch hỗ trợ CUDA theo
[hướng dẫn chính thức](https://pytorch.org/get-started/locally/). Nếu cần bản khác
với torch/torchvision trong requirements, điều chỉnh cặp phiên bản phù hợp trước
khi cài và lưu lại môi trường đã kiểm thử bằng `python -m pip freeze`.

Kiểm tra thiết bị:

```bash
python - <<'PY'
import sys, torch, ultralytics
print('Python:', sys.executable)
print('Ultralytics:', ultralytics.__version__)
print('PyTorch:', torch.__version__)
print('MPS:', torch.backends.mps.is_available())
print('CUDA:', torch.cuda.is_available())
PY
```

| Device | Khi dùng |
|---|---|
| `auto` | Tự chọn CUDA, rồi MPS, cuối cùng CPU |
| `mps` | Mac có backend MPS khả dụng |
| `0` | GPU NVIDIA thứ nhất, cần CUDA khả dụng |
| `cpu` | Kiểm tra chức năng hoặc máy không có GPU phù hợp |

Nếu chạy trong môi trường hạn chế quyền truy cập GPU, kết quả kiểm tra có thể
khác với terminal trên máy. Không thay `mps` bằng `0` khi không có CUDA.

Các lần mở terminal sau:

```bash
cd /Users/nguyenson/Tuankiet
source ai/.venv/bin/activate
```

## 2. Dữ liệu và lớp

Dataset gốc ở `ai/data/raw/CarDD/`, dữ liệu polygon YOLO ở `ai/data/yolo/`.
Giữ cả hai để đối chiếu. Không chia lại split chính thức:

| Split | Ảnh | Annotation |
|---|---:|---:|
| train | 2816 | 6211 |
| val | 810 | 1744 |
| test | 374 | 785 |

| ID YOLO | Tên model | Enum JSON |
|---:|---|---|
| 0 | dent | DENT |
| 1 | scratch | SCRATCH |
| 2 | crack | CRACK |
| 3 | glass_shatter | CRACK |
| 4 | lamp_broken | CRACK |
| 5 | tire_flat | Lọc khỏi JSON nghiệp vụ |

Vẫn train/evaluate đủ 6 lớp. Helper inference nhận cả tên có `_` và tên có dấu
cách; JSON của predictor dùng tên có dấu cách để truy vết. ID không đổi.
`TEAR_PUNCTURE` và `PAINT_CHIPPING` chưa có lớp train tương ứng.

Thống kê và vẽ kiểm tra nhãn vẫn chạy từ thư mục `ai/`:

```bash
cd /Users/nguyenson/Tuankiet/ai
python scripts/stats_cardd.py
python scripts/visualize_labels.py
cd /Users/nguyenson/Tuankiet
```

Xem ảnh trong `ai/reports/overlay/`. Visualizer cần ít nhất 20 ảnh, chọn mẫu ngẫu
nhiên và vẽ polygon, không phải bbox. Mỗi lần chạy có thể để lại cả ảnh mẫu cũ.

## 3. Convert — chỉ khi cần tạo lại dữ liệu

Dataset đã convert sẵn, nên bỏ qua bước này khi chỉ train hoặc dự đoán:

```bash
python ai/scripts/coco_to_yolo.py
```

Converter hiện resolve đường dẫn theo vị trí script. Nó ghi nhãn polygon vào
`ai/data/yolo/labels/{train,val,test}/`, copy ảnh và ghi lại `ai/configs/data.yaml`.
Nhãn cùng tên bị ghi lại; ảnh đã có được giữ. Chưa dọn output thừa và chưa dừng
bằng exception khi thiếu ảnh; cần xem log `thiếu file 0`, `bỏ qua 0 polygon`.
Không xóa nhãn tire flat.

Mỗi dòng segmentation: `class_id x1 y1 x2 y2 ...`, tọa độ chuẩn hóa 0–1.
Dùng model `-seg.pt`, không đổi sang checkpoint classification/detection.

Train/evaluate tạo YAML riêng cho lượt chạy, resolve path và kiểm tra đủ số ảnh
cùng cặp nhãn trên cả ba split. Với YAML tương đối, `path` được tính từ thư mục
`ai/`. Với YAML tuyệt đối trên máy cũ, phải sửa `path` khi chuyển sang máy khác.

## 4. Train từ model nền

```bash
python ai/train/train_damage.py --epochs 30 --imgsz 640 --batch 8 --device auto --workers 0 --patience 10 --name cardd_baseline30
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--model` | `ai/yolov8n-seg.pt` được resolve tuyệt đối | Model nền hoặc checkpoint fine-tune |
| `--data` | `ai/configs/data.yaml` | Cấu hình 6 lớp |
| `--epochs` | 30 | Số epochs tối đa của lượt chạy |
| `--imgsz` | 640 | Kích thước ảnh |
| `--batch` | 8 | Số ảnh mỗi batch |
| `--device` | auto | Thiết bị |
| `--workers` | 0 | Worker đọc dữ liệu |
| `--patience` | 10 | Early stopping theo validation |
| `--name` | cardd_30epochs | Tên thư mục thí nghiệm, phải là tên mới |
| `--seed` | 0 | Seed |

Kết quả mới được lưu trực tiếp trong `ai/runs/damage/<name>/`.
Không dùng lại tên đã tồn tại. Dữ liệu train/val vẫn giữ nguyên, không đọc test
để chọn checkpoint. Ultralytics lưu `weights/best.pt` theo fitness validation
của thư viện; `last.pt` là checkpoint cuối được lưu.

5 epochs chỉ phù hợp chạy thử pipeline; số epochs lớn hơn cũng không bảo đảm
đạt 0.80. Khi hết bộ nhớ, giảm batch xuống 4, 2 hoặc 1 trước. Chỉ thử imgsz 960/1024
khi baseline ổn định và đánh giá thay đổi bằng validation.

## 5. Train tiếp từ checkpoint cũ

Lệnh fine-tune đang dùng cho đợt cải thiện đầu tiên:

```bash
python ai/train/train_damage.py --model ai/runs/segment/runs/damage/smoke/weights/best.pt --epochs 10 --imgsz 640 --batch 16 --device mps --workers 0 --patience 5 --name cardd_finetune10_v2
```

Lệnh trên minh họa lượt chạy đã khởi tạo. Nếu chạy lại, dùng tên khác vì folder
đã tồn tại. Chọn device/batch theo máy thực tế. Đây là fine-tune một lượt mới từ
weights cũ, không phải resume nguyên optimizer và scheduler của lượt trước.

Tên lớp trong YAML được giữ ở dạng underscore như checkpoint cũ để thư viện nạp
đủ các đầu ra theo tên; inference sẽ chuẩn hóa khi xuất JSON. Lượt
`cardd_finetune10` ban đầu đã bị dừng để sửa điểm này, không dùng nó để bàn giao.

## 6. Theo dõi kết quả training

```bash
find ai/runs -type f -name best.pt
find ai/runs -type f -name results.csv
```

Lượt cũ ở `ai/runs/segment/runs/damage/smoke/`; lượt mới ở
`ai/runs/damage/cardd_finetune10_v2/`. Không tự di chuyển checkpoint giữa lúc train.

| File | Nội dung |
|---|---|
| `args.yaml` | Tham số thực tế |
| `dataset.yaml` | Dataset đã resolve cho lượt chạy mới |
| `results.csv` | Loss và metric theo epoch |
| `results.png` | Đồ thị training |
| `weights/best.pt` | Model tốt nhất theo validation fitness của thư viện |
| `weights/last.pt` | Model cuối |
| `val_batch*_pred.jpg` | Ảnh dự đoán validation |

Log fine-tune hiện tại: `ai/reports/train_finetune10_v2.log`.
Các metric `(B)` là bounding box, `(M)` là mask. `mAP50` khác `mAP50-95`.

## 7. Đánh giá từng checkpoint, từng lớp

Ví dụ tạo báo cáo validation cho một model mới. Chọn thư mục output mới:

```bash
python ai/scripts/evaluate_damage.py --model ai/runs/damage/cardd_baseline30/weights/best.pt --split val --output ai/reports/metrics/baseline30_val --device auto --imgsz 640 --batch 8
```

Đầu ra gồm `metrics.json`, `per_class.csv`, `dataset.yaml` và thư mục biểu đồ của
thư viện. CSV ghi precision, recall, AP@50 và AP@50:95 cho box/mask của từng lớp.
Lớp không có kết quả AP được ghi null/ô trống thay vì lấy mAP tổng làm AP lớp đó.
Báo cáo có SHA256 checkpoint để biết chính xác weights đã đánh giá.

Không dùng test để quyết định epochs, imgsz hay chọn checkpoint. Khi đổi model
hoặc imgsz, dùng output mới để không trộn báo cáo.

## 8. Chọn model bằng val rồi đánh giá test riêng

Sau khi fine-tune kết thúc:

```bash
python ai/scripts/finalize_damage.py --baseline ai/runs/segment/runs/damage/smoke/weights/best.pt --candidate ai/runs/damage/cardd_finetune10_v2/weights/best.pt --device cpu --imgsz 640
```

Quy trình tự động:

1. Đánh giá baseline và candidate trên val cùng imgsz.
2. So sánh mask mAP50-95; nếu bằng nhau giữ baseline.
3. Ghi checkpoint đã chọn vào manifest trước khi chạy test.
4. Chỉ đánh giá test cho model được chọn; không dùng test để đổi lựa chọn.
5. Chạy thử một ảnh validation qua CLI JSON.

Đầu ra:

- `ai/reports/metrics/baseline_val/`: baseline trên val.
- `ai/reports/metrics/finetune_val/`: fine-tune trên val.
- `ai/reports/metrics/selected_test/`: model đã chọn trên test.
- `ai/reports/model_selection.json`: model được chọn, hash, cấu hình và metric.
- `ai/reports/KET_QUA.md`: bảng tóm tắt để review.
- `ai/reports/inference_selected.json`: một ví dụ suy luận.

Script tái sử dụng báo cáo chỉ khi hash checkpoint, split và imgsz còn khớp.
Nếu dùng thí nghiệm khác mà những folder cố định đã có báo cáo cũ, cần đổi vị trí
báo cáo trong script hoặc lưu riêng bộ báo cáo cũ trước; script sẽ báo lỗi thay vì
lặng lẽ dùng số liệu của model khác.

## 9. Nhận ảnh và xuất JSON bằng CLI

Ví dụ dùng checkpoint smoke đã có để kiểm tra chức năng ngay:

```bash
python ai/inference/infer.py --model ai/runs/segment/runs/damage/smoke/weights/best.pt --source ai/data/yolo/images/val/000016.jpg --output ai/reports/prediction.json --device auto --imgsz 640 --conf 0.25
```

Để dùng model được chọn sau bước 8, lấy đường dẫn từ manifest:

```bash
export CARDD_MODEL="$(python -c "import json; print(json.load(open('ai/reports/model_selection.json'))['selected_checkpoint'])")"
python ai/inference/infer.py --model "$CARDD_MODEL" --source ai/data/yolo/images/val/000016.jpg --output ai/reports/prediction_selected.json --device auto
```

`--source` nhận một file hoặc một thư mục ảnh; thư mục được duyệt cả thư mục con.
Các ảnh cần decode được bằng OpenCV. Nếu một ảnh lỗi, lệnh dừng và không ghi bộ
JSON hoàn chỉnh. `--output` sẽ được ghi lại nếu đã tồn tại.

Ví dụ cấu trúc đầu ra, các con số dưới đây chỉ minh họa schema:

```json
{
  "schema_version": "1.0",
  "model": "/path/to/best.pt",
  "confidence_threshold": 0.25,
  "imgsz": 640,
  "coordinate_system": "pixels; bbox_xyxy=[left,top,right,bottom]",
  "images": [{
    "image": "/path/to/car.jpg",
    "width": 1000,
    "height": 667,
    "detections": [{
      "class_id": 3,
      "class_name": "glass shatter",
      "damage_type": "CRACK",
      "confidence": 0.85,
      "bbox_xyxy": [10, 20, 180, 220],
      "polygon_xy": [[10,20], [180,20], [180,220]]
    }],
    "excluded_tire_flat": 0
  }]
}
```

Tọa độ tính bằng pixel trên ảnh gốc. `excluded_tire_flat` đếm kết quả tire flat
đã phát hiện nhưng bị lọc khỏi output nghiệp vụ. Không phát hiện không có nghĩa
xe chắc chắn không hỏng. Model chỉ hỗ trợ các lớp đã train; không tự suy ra chi phí
sửa chữa, mức độ hư hỏng hoặc hai enum chưa có dữ liệu.

## 10. API local nhận ảnh upload

Từ thư mục gốc project, đặt checkpoint rồi chạy một worker:

```bash
export CARDD_MODEL="/Users/nguyenson/Tuankiet/ai/runs/segment/runs/damage/smoke/weights/best.pt"
export CARDD_DEVICE="auto"
python -m uvicorn ai.inference.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Sau bước 8, có thể bỏ biến CARDD_MODEL để API tự lấy manifest, hoặc đặt rõ đường
checkpoint đã chọn. Không dùng model nền COCO: API kiểm tra task và đủ 6 lớp CarDD.
Đường dẫn tương đối trong biến môi trường được tính từ thư mục gọi server.

Mở `http://127.0.0.1:8000/docs` để chọn file và thử `POST /predict`.
Terminal thứ hai có thể gọi:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -X POST http://127.0.0.1:8000/predict -F "file=@/Users/nguyenson/Tuankiet/ai/data/yolo/images/val/000016.jpg"
```

API trả cùng envelope JSON như CLI. Model được nạp khi khởi động và tái sử dụng;
các lần suy luận được khóa để tránh gọi đồng thời cùng một model. Ảnh upload
được lưu tạm riêng từng request rồi dọn; giới hạn 20 MiB mỗi file. File trống,
thiếu hoặc không decode được trả 422; quá giới hạn trả 413. Đây là API local để
tích hợp thử, chưa cấu hình xác thực và triển khai public.

Ví dụ phản hồi qua model thật: `ai/reports/api_smoke.json`.

## 11. Kiểm tra ảnh thực tế ngoài CarDD

Đặt ảnh xe ngoài dataset vào một thư mục riêng, ví dụ `ai/data/external/`.
Không dùng ảnh CarDD đổi tên để gọi là kiểm tra ngoài dataset.

```bash
python ai/inference/infer.py --model "$CARDD_MODEL" --source ai/data/external --output ai/reports/inference_external.json --device auto
```

Nếu muốn bước 8 chạy thêm ảnh này, truyền `--external ai/data/external`.
Phải có ảnh thực tế được cung cấp trước; project hiện chưa xác nhận có bộ ảnh như vậy.
Không có ground truth thì chỉ review trực quan, chưa thể tính AP/độ chính xác.
Để đo AP, cần nhãn polygon cho tập ngoài và cấu hình đánh giá riêng; script
`evaluate_damage.py` hiện cố ý giới hạn ở split CarDD chính thức.

## 12. Chạy tests và xử lý lỗi

```bash
python -m pip install -r ai/requirements-dev.txt
python -m unittest discover -s ai/tests -v
```

Tests kiểm tra mapping tất cả alias, lọc tire flat, không sửa input, schema
bbox/polygon, ảnh lỗi, model sai lớp/task và API upload. Tests không train model.
Đã có kiểm tra tích hợp riêng bằng weights thật và ảnh validation.

| Lỗi | Cách xử lý |
|---|---|
| Thiếu package | Kích hoạt đúng venv, cài requirements bằng cùng Python |
| Không tìm thấy dataset trên máy khác | Sửa `path` trong configs/data.yaml |
| `Run already exists` | Đổi `--name`, giữ run cũ để so sánh |
| `Report exists` hoặc `Stale report` | Dùng output mới hoặc lưu riêng báo cáo cũ |
| Không đủ ảnh/thiếu label | Kiểm tra lại dataset, không bỏ qua lỗi rồi train |
| MPS/CUDA không khả dụng | Kiểm tra thiết bị thực, dùng cpu khi cần |
| Hết bộ nhớ | Giảm batch; cân nhắc giảm imgsz |
| Model không phải 6 lớp CarDD | Dùng best.pt đã train, không dùng model COCO nền |
| API khởi động lỗi | Kiểm tra CARDD_MODEL và CARDD_DEVICE, hoặc manifest đã chọn model |
| Visualizer báo sample lớn hơn số ảnh | Chạy từ ai/ và kiểm tra đã có đủ ảnh |

## Tham chiếu

- [Ultralytics training](https://docs.ultralytics.com/modes/train)
- [Segmentation, box/mask metrics và predict](https://docs.ultralytics.com/tasks/segment)
- [FastAPI upload file](https://fastapi.tiangolo.com/tutorial/request-files/)
- [PyTorch theo nền tảng](https://pytorch.org/get-started/locally/)
