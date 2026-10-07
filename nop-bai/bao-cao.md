# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Cao Văn Cường |
| MSSV | 2A202602493 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/caovancuong611/K4-L3-DAY21-CaoVanCuong-2A202602493-CI-CD-for-AI-Systems |
| Ngày nộp | ___ |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 4 | 300 | 0.05 | 4 | 0.7070 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ này có f1_score cao nhất (0.7149) trên tập holdout và vượt ngưỡng 0.65. Lần chạy có
accuracy cao nhất lại là lần 1 (0.8780), không trùng với lần có F1 cao nhất. Điều đó cho thấy
accuracy chủ yếu phản ánh lớp đa số, còn F1 mới phản ánh khả năng nhận ra người thu nhập cao.
Lần 2 dùng ít cây, learning_rate nhỏ và cây nông nên chưa học đủ: F1 chỉ 0.6051 (dưới ngưỡng),
trong khi accuracy vẫn ở mức 0.846. Lần 4 cho thấy đánh đổi giữa n_estimators và learning_rate:
giảm learning_rate xuống 0.05 thì cần tới 300 cây mới đạt F1 gần bằng cấu hình 200 cây với
learning_rate 0.1.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập trên 50K, nên dữ liệu mất cân bằng. Một mô hình luôn trả lời
"thu nhập thấp" vẫn đạt accuracy 0,752 dù không phát hiện được người thu nhập cao nào, và
F1 của nó bằng 0. Vì vậy accuracy cao không chứng minh được mô hình hữu ích. Ở Bước 1, accuracy
của các lần chạy chỉ dao động từ 0,846 đến 0,878, còn F1 dao động từ 0,605 đến 0,715. F1 của lớp
dương là trung bình điều hòa của precision và recall trên lớp thiểu số, nên nó giảm mạnh khi mô
hình bỏ sót hoặc gán nhầm người thu nhập cao. Không dùng `average="weighted"` hay `"macro"` vì
các cách tính này gộp cả F1 của lớp đa số (khoảng 0,92), làm điểm số bị đẩy lên và ngưỡng 0,65
mất tác dụng chặn mô hình kém.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Python mặc định trên máy là 3.14, không cài được các thư viện đã pin. | Các phiên bản trong `requirements.txt` chỉ hỗ trợ Python 3.10 - 3.12. | Tạo `.venv` bằng `py -3.12 -m venv .venv`. |
| `import mlflow` báo lỗi `No module named 'pkg_resources'`. | pip kéo về setuptools 84, bản này đã bỏ `pkg_resources` mà MLflow 2.13 cần. | Pin `setuptools<81` trong `requirements.txt`. |
| MLflow với `sqlite:///mlflow.db` báo lỗi `FallbackAsyncAdaptedQueuePool`. | SQLAlchemy 2.1 không tương thích với MLflow 2.13. | Pin `sqlalchemy<2.1` trong `requirements.txt`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: chưa thực hiện.
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: quét ngưỡng 0.10 - 0.90; với bộ đã chọn, ngưỡng 0.30 cho F1 0.7368 so với 0.7149 ở ngưỡng 0.5, vì hạ ngưỡng giúp tăng recall của lớp thiểu số.
- [x] Bonus 3 - Báo cáo precision / recall tự động: `outputs/detail.txt` (confusion matrix + precision/recall từng lớp) được in ra log CI và upload artifact; bỏ sót người thu nhập cao (recall lớp 1 = 0.64) là sai lầm đáng chú ý hơn vì đó là nhóm bài toán cần tìm.
- [x] Bonus 4 - Hoàn trả về phiên bản trước: Quality Gate so sánh F1 mới với `artifacts/current/report.json`; đặt biến `BLOCK_ON_F1_REGRESSION=true` để chặn Release khi F1 giảm.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ghi `train_positive_rate` vào `report.json`/MLflow và in `WARNING: DATA DRIFT` khi lệch quá 5 điểm % so với 24,8%.
