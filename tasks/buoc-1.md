# Bước 1 - Thực Nghiệm Cục Bộ và Theo Dõi Thí Nghiệm

Mục tiêu: Chạy ít nhất 3 thí nghiệm huấn luyện với các siêu tham số khác nhau. So sánh kết quả trong MLflow UI. Xác định bộ siêu tham số tốt nhất để sử dụng ở Bước 2.

---

## 1.1 Tải Dữ Liệu

Chạy script đã được cung cấp sẵn (không cần sửa):

```bash
python prepare_data.py
```

Kết quả mong đợi:

```
train_batch1.csv : 22361 mau
holdout.csv      : 500 mau
train_batch2.csv : 22361 mau
Ty le lop >50K   : 24.8%
```

Xác nhận các file đã được tạo:

```bash
ls data/
```

---

## 1.2 Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

---

## 1.3 Cấu Hình MLflow

MLflow sử dụng SQLite làm backend lưu trữ cục bộ. Thêm hai biến môi trường sau vào shell hoặc file `.env`:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

Trên Windows PowerShell, dùng:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "./mlartifacts"
```

Không cần khởi động server riêng. MLflow sẽ ghi dữ liệu thí nghiệm vào file cục bộ `mlflow.db`.

---

## 1.4 Hiểu Bài Toán Trước Khi Viết Code

Đọc kỹ phần này trước khi làm tiếp — nó quyết định cách bạn đánh giá mô hình trong suốt lab.

Tập dữ liệu Adult có phân bố lớp **mất cân bằng**: chỉ 24,8% số mẫu thuộc lớp thu nhập cao. Hệ quả:

```
Mô hình "luôn trả lời thu nhập thấp":
    accuracy = 0.752      <- trông rất cao
    f1_score = 0.000      <- không bắt được một trường hợp thu nhập cao nào
```

Accuracy 0,752 nghe có vẻ tốt, nhưng mô hình đó hoàn toàn vô dụng. Vì vậy:

- Chỉ số chính của lab này là `f1_score` **của lớp dương** (target = 1, thu nhập > 50K).
- Hàm `train()` trả về `f1`, không phải accuracy.
- Ngưỡng chất lượng ở Bước 2 là `f1_score >= 0.65`.
- Accuracy vẫn được ghi lại để tham khảo, nhưng không dùng làm căn cứ chặn triển khai.

Lưu ý khi gọi `f1_score`: dùng `f1_score(y_eval, preds)` — mặc định tính cho lớp dương. **Không** truyền `average="weighted"` hay `average="macro"`, vì các giá trị đó bị lớp đa số kéo lên cao và làm mất ý nghĩa của ngưỡng.

---

## 1.5 Viết `params.yaml`

Tạo file `params.yaml` ở thư mục gốc của project. File này chứa các siêu tham số cho mô hình GradientBoosting. Bạn sẽ thay đổi các giá trị này giữa các lần chạy để so sánh hiệu quả.

```yaml
n_estimators: 100
learning_rate: 0.1
max_depth: 3
```

Giải thích từng tham số:

| Tham số | Ý nghĩa | Gợi ý giá trị thử nghiệm |
|---|---|---|
| n_estimators | Số cây được cộng dồn qua các vòng boosting | 50, 100, 200 |
| learning_rate | Mức đóng góp của mỗi cây vào kết quả cuối | 0.05, 0.1, 0.2 |
| max_depth | Độ sâu tối đa của mỗi cây | 2, 3, 5 |

Khác với RandomForest (các cây độc lập nhau), GradientBoosting huấn luyện cây sau để sửa lỗi của các cây trước. Vì vậy `n_estimators` và `learning_rate` có quan hệ đánh đổi: giảm `learning_rate` thì thường phải tăng `n_estimators` để bù lại.

---

## 1.6 Huấn Luyện Mô Hình

`src/train.py` đã được cài đặt sẵn. Script đọc `train_batch1.csv`, huấn luyện
`GradientBoostingClassifier` bằng các giá trị trong `params.yaml`, ghi `f1_score` và
`accuracy` vào MLflow, đồng thời lưu model và `outputs/report.json`. Chạy script từ thư
mục gốc của repo:

```bash
python src/train.py
```

---

## 1.7 Chạy Ít Nhất 3 Thí Nghiệm

Chỉnh sửa `params.yaml` giữa mỗi lần chạy để thay đổi siêu tham số. Ví dụ:

```bash
# Lần 1: giá trị mặc định
python src/train.py

# Chỉnh sửa params.yaml -> n_estimators: 50, learning_rate: 0.05, max_depth: 2
python src/train.py

# Chỉnh sửa params.yaml -> n_estimators: 200, learning_rate: 0.1, max_depth: 5
python src/train.py
```

Gợi ý: Chạy thêm 1-2 lần nữa với các giá trị khác để có nhiều dữ liệu so sánh hơn.

Quan sát quan trọng: hãy chú ý cột accuracy và cột f1_score biến động khác nhau như thế nào giữa các lần chạy. Bạn sẽ thấy accuracy gần như không đổi trong khi f1_score chênh lệch rõ rệt — đó chính là lý do lab này chọn F1 làm chỉ số quyết định.

---

## 1.8 Phân Tích Kết Quả Trên MLflow UI

Khởi động MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Truy cập http://localhost:5000. Bạn sẽ thấy tất cả các lần chạy được liệt kê.

Trong giao diện MLflow UI, hãy:

1. Sắp xếp các lần chạy theo `f1_score` (giảm dần) để tìm lần chạy tốt nhất.
2. Chọn nhiều lần chạy và nhấn "Compare" để xem biểu đồ so sánh.
3. Ghi nhận bộ siêu tham số của lần chạy có `f1_score` cao nhất.
4. Kiểm tra xem lần chạy có accuracy cao nhất có phải cũng là lần có f1_score cao nhất không. Nếu không, hãy ghi lại nhận xét này vào báo cáo.

Đặt bộ siêu tham số tốt nhất vào `params.yaml` trước khi chuyển sang Bước 2.

Lưu ý: bộ siêu tham số bạn chọn phải đạt `f1_score >= 0.65`, nếu không pipeline ở Bước 2 sẽ chặn bước triển khai.

---

## Kết Quả Cần Đạt - Bước 1

Trước khi chuyển sang Bước 2, kiểm tra các điểm sau:

- `src/train.py` chạy thành công không có lỗi.
- File `outputs/report.json` tồn tại và chứa cả `f1_score` và `accuracy`.
- File `models/model.joblib` tồn tại.
- MLflow UI hiển thị ít nhất 3 lần chạy với các siêu tham số khác nhau.
- `params.yaml` đã được cập nhật với bộ siêu tham số tốt nhất, đạt `f1_score >= 0.65`.

Chụp màn hình MLflow UI và lưu thành `nop-bai/anh-chup-man-hinh/01-mlflow-ui.png`
(yêu cầu chi tiết của ảnh: [nop-bai/anh-chup-man-hinh/README.md](../nop-bai/anh-chup-man-hinh/README.md)).

Ghi ngay bộ siêu tham số tốt nhất và các số liệu so sánh vào mục 1 của
[nop-bai/bao-cao.md](../nop-bai/bao-cao.md) — làm lúc này dễ hơn nhiều so với việc nhớ lại
sau khi đã hoàn thành cả ba bước.

---

Tiếp theo: [Bước 2 - Pipeline CI/CD tự động](buoc-2.md)
