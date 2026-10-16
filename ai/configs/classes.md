# Bảng lớp CarDD và mapping output

Train đủ 6 lớp gốc. Chỉ gộp/lọc khi xuất JSON trong `inference/infer.py`.

| ID COCO | ID YOLO | Lớp train | Enum output |
|---:|---:|---|---|
| 1 | 0 | dent | DENT |
| 2 | 1 | scratch | SCRATCH |
| 3 | 2 | crack | CRACK |
| 4 | 3 | glass shatter | CRACK |
| 5 | 4 | lamp broken | CRACK |
| 6 | 5 | tire flat | Bỏ khỏi output |

`TEAR_PUNCTURE` và `PAINT_CHIPPING` được khai báo trong enum, nhưng chưa có
lớp train tương ứng. Cần bổ sung ảnh thực tế và nhãn trước khi huấn luyện hai lớp này.

## Comment để review

- Thứ tự lấy từ `categories` của cả ba annotation đang có trong dataset.
  Khi convert, dùng `category_id - 1`; không sắp xếp lớp theo alphabet.
- Không xóa nhãn hoặc ảnh tire flat khi convert/train/evaluate, nhằm tránh
  biến vật thể đã có nhãn thành background. Lọc ở bước xuất JSON.
- Gộp glass shatter và lamp broken vào CRACK là quy ước nghiệp vụ theo bảng
  yêu cầu, không khẳng định chúng là cùng loại hư hỏng về mặt thị giác.
- JSON helper giữ `class_name` gốc và thêm `damage_type` để truy vết.
  Đây là định dạng trung gian để review; chưa xác nhận schema API toàn bộ với SRS.
- Tên lớp lạ gây lỗi thay vì âm thầm bỏ kết quả. Chỉ tire flat được bỏ có chủ đích.

## Split và đánh giá

| Split chính thức | Số ảnh | Số annotation |
|---|---:|---:|
| train | 2816 | 6211 |
| val | 810 | 1744 |
| test | 374 | 785 |

Giữ nguyên split khi chuyển sang YOLO. Chỉ cần chia theo xe khi bổ sung dữ liệu
ảnh thực tế của nhóm. Dùng val để chọn model/tham số, giữ test cho đánh giá cuối.

Báo cáo AP theo từng lớp gốc và mAP tổng, ghi rõ AP@0.50 và AP@0.50:0.95.
Đánh giá trên 6 lớp trước khi mapping; kết quả trên nhóm enum đã gộp cần báo riêng.
Mục tiêu 0.80 cần xác định metric cụ thể, chưa có kết quả train để xác nhận đạt.
Theo đề xuất, crack nhỏ và scratch mảnh cần được theo dõi riêng. Nếu AP crack thấp,
thử imgsz 960 hoặc 1024 và đo lại trên val; điều chỉnh batch theo VRAM thực tế.

## Phạm vi đã triển khai

Đã có converter YOLO segmentation, train, báo cáo AP từng lớp, chọn checkpoint
bằng validation và đánh giá test riêng. `inference/infer.py` nhận ảnh hoặc thư mục
ảnh để xuất JSON; `inference/api.py` nhận file upload qua API local.

Tên model giữ dạng underscore để tương thích weights cũ. Mapping hỗ trợ cả
underscore và dấu cách; ID và đủ 6 lớp không đổi. Xem `reports/model_selection.json`
và `reports/KET_QUA.md` sau khi hoàn tất đánh giá để biết model và số liệu thực tế.
Ảnh ngoài CarDD cần được cung cấp riêng; không tự xem ảnh dataset là ảnh kiểm tra thực tế.
