# Bước 2 - Pipeline CI/CD trên AWS

Mục tiêu: GitHub Actions kiểm thử, lấy dữ liệu DVC từ Amazon S3, huấn luyện mô hình,
chặn mô hình có `f1_score < 0.65`, rồi triển khai API lên EC2 nếu đạt chất lượng.

> **Chi phí:** dùng AWS Free plan, không chọn **Upgrade plan**. Quyền truy cập Free plan
> kết thúc khi hết thời hạn hoặc credit theo thông tin Billing của tài khoản. Không phải
> mọi dịch vụ/loại EC2 đều khả dụng trong Free plan. Nếu AWS yêu cầu nâng cấp hoặc hiện
> thông báo giá trước khi tạo tài nguyên, dừng lại và không xác nhận. Cảnh báo chi phí
> không tự dừng tài nguyên.

## 2.1 Chọn Region và tạo S3 bucket

Chọn một region gần bạn và dùng thống nhất cho bucket, EC2 và GitHub Actions. Ví dụ dưới
đây dùng `ap-southeast-1`; thay bằng region bạn chọn. Tên S3 bucket phải duy nhất toàn
cầu, viết thường, ví dụ `income-lab-<chuoi-ngau-nhien>`.

Trong AWS Console:

1. Mở **S3 → Create bucket**.
2. Chọn region đã định; giữ **Block all public access** bật.
3. Tạo bucket, sau đó ghi lại tên bucket và region.
4. Chưa bật versioning, replication hay logging cho lab này để tránh tạo thêm tài nguyên.

## 2.2 Cấp quyền AWS an toàn

Không tạo access key cho root account. Dùng các danh tính riêng theo từng mục đích:

| Danh tính | Mục đích | Quyền cần có |
|---|---|---|
| Máy cá nhân (IAM user/Identity Center) | `dvc push` và `dvc pull` | Đọc/ghi/xóa object dưới `dvc/` trong bucket lab |
| GitHub Actions IAM role | CI kéo dữ liệu, phát hành model và mở SSH tạm thời | Đọc `dvc/*`; đọc/ghi `artifacts/*`; authorize/revoke SSH ingress trên đúng security group |
| EC2 instance role | API tải model khi khởi động | Chỉ `s3:GetObject` với `artifacts/current/model.joblib` |

Với GitHub Actions, ưu tiên **OIDC role**, không lưu access key dài hạn trong GitHub Secrets:

1. Trong **IAM → Identity providers**, thêm OpenID Connect provider
   `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. Tạo IAM role cho GitHub Actions với trusted entity là Web identity. Trust policy cần
   giới hạn `sub` vào repo của bạn và nhánh `main`, dạng
   `repo:OWNER/REPO:ref:refs/heads/main`, cùng `aud` là `sts.amazonaws.com`.
3. Gắn policy giới hạn bucket lab như bảng trên, cùng `ec2:AuthorizeSecurityGroupIngress`
   và `ec2:RevokeSecurityGroupIngress` chỉ trên security group của lab. Không cấp
   `AdministratorAccess`.
4. Ghi lại ARN role để lưu thành GitHub Secret `AWS_ROLE_ARN`.

Tạo IAM instance profile/role cho EC2 với quyền đọc đúng object model nêu trên và gắn
role đó vào EC2. Không chép AWS access key lên VM; SDK trên EC2 sẽ dùng instance role.

Để DVC từ máy cá nhân truy cập S3, dùng danh tính riêng có quyền bucket-scoped. Cài AWS
CLI theo [hướng dẫn chính thức](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
và cấu hình AWS profile cục bộ theo hướng dẫn của AWS. Lưu credentials ngoài repo, không
dùng root access key, không gửi key/secret/OTP qua chat, và xóa key nếu không còn cần
dùng.

## 2.3 Theo dõi dữ liệu với DVC và S3

Cài các dependencies của repo trước:

```bash
python -m pip install -r requirements.txt
dvc init
dvc remote add -d labstore s3://<TEN_BUCKET>/dvc
dvc remote modify labstore region <AWS_REGION>

dvc add data/train_batch1.csv
dvc add data/holdout.csv
dvc add data/train_batch2.csv
git add .gitignore .dvc/config data/*.dvc
git commit -m "feat: track datasets with DVC"
dvc push
```

Commit file con trỏ `.dvc` và `.dvc/config`; **không commit CSV hay AWS credentials**.
Trong S3 Console, xác nhận object dữ liệu nằm dưới prefix `dvc/`.

## 2.4 Tạo EC2 và chuẩn bị API

1. Mở **EC2 → Launch instance**. Chọn Ubuntu LTS và loại instance được Console đánh dấu
   phù hợp với Free plan của chính tài khoản/region bạn. Nếu không có loại phù hợp thì
   dừng lại; không nâng cấp chỉ để tiếp tục lab.
2. Tạo key pair mới, tải private key về máy và giữ riêng. Gắn IAM instance role đã tạo ở
   mục 2.2.
3. Security group: chỉ mở SSH (22) từ **My IP** để bạn cấu hình máy; mở TCP 8080 từ IP
   của bạn để kiểm thử API. Không mở `0.0.0.0/0`. Workflow sẽ tạm thêm địa chỉ IP
   `/32` của GitHub runner vào SSH ingress và xóa rule ngay cả khi bước deploy lỗi.
4. Sau khi instance chạy, ghi lại **Public IPv4 address** và SSH username Ubuntu thường
   là `ubuntu`.

SSH vào EC2, cài môi trường cho API (không cần cài toàn bộ DVC/MLflow trên VM):

```bash
sudo apt update
sudo apt install -y python3-venv
mkdir -p ~/income-api/src ~/income-api/models
python3 -m venv ~/income-api/venv
~/income-api/venv/bin/pip install \
  fastapi==0.111.0 uvicorn==0.29.0 scikit-learn==1.4.2 \
  joblib==1.4.2 boto3==1.34.131 numpy==1.26.4
```

Tạo systemd unit `/etc/systemd/system/income-api.service` bằng `sudo nano`:

```ini
[Unit]
Description=Income Model Inference API
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/income-api
Environment="ARTIFACT_BUCKET=<TEN_BUCKET>"
Environment="AWS_DEFAULT_REGION=<AWS_REGION>"
ExecStart=/home/ubuntu/income-api/venv/bin/uvicorn src.serve:app --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Nếu SSH username khác `ubuntu`, đổi cả `User` và đường dẫn `/home/ubuntu`. Thay
`<TEN_BUCKET>` bằng bucket thật. Sau đó:

```bash
sudo systemctl daemon-reload
sudo systemctl enable income-api
```

Service sẽ được khởi động sau lần Actions đầu tiên upload model. Không tải service từ
S3 bằng access key trên EC2; instance role ở mục 2.2 cấp quyền tải model.

## 2.5 SSH key cho GitHub Actions

Trên máy cá nhân (Git Bash, Linux hoặc macOS), tạo **deploy key riêng** (không dùng lại EC2 key pair):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/income_deploy -N "" -C "github-actions-deploy"
```

Đăng nhập EC2 bằng key pair lúc tạo máy:

```bash
ssh -i <EC2_KEY_PAIR.pem> ubuntu@<EC2_PUBLIC_IP>
```

Mở `~/.ssh/authorized_keys` trên EC2 bằng `nano`, thêm một dòng là nội dung file
`income_deploy.pub`, rồi đặt quyền:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Giữ file `income_deploy` (không có `.pub`) private ở máy để đưa trực tiếp vào GitHub
Secret. Không commit private key.

## 2.6 Cấu hình GitHub Actions

Trong repo GitHub vào **Settings → Secrets and variables → Actions**. Tạo:

**Repository secrets**

| Tên | Giá trị |
|---|---|
| `AWS_ROLE_ARN` | ARN OIDC role của GitHub Actions |
| `ARTIFACT_BUCKET` | Tên S3 bucket |
| `SERVER_SECURITY_GROUP` | ID security group EC2 (dạng `sg-...`) để mở SSH tạm thời cho runner |
| `SERVER_HOST` | Public IPv4/DNS của EC2 |
| `SERVER_USER` | SSH user (thường `ubuntu`) |
| `SERVER_SSH_KEY` | Toàn bộ private deploy key `income_deploy` |

**Repository variable**

| Tên | Giá trị |
|---|---|
| `AWS_REGION` | Region dùng cho S3 và IAM role, ví dụ `ap-southeast-1` |
| `BLOCK_ON_F1_REGRESSION` | (Tùy chọn, Bonus 4) đặt `true` để chặn Release khi F1 mới thấp hơn model đang chạy; bỏ trống thì chỉ cảnh báo |

Workflow đã cấu hình OIDC và bốn jobs Unit Test → Train → Quality Gate → Release.
Train tải dữ liệu, ghi model candidate vào `artifacts/candidates/<commit>/`; model chỉ
được copy sang `artifacts/current/model.joblib` sau khi quality gate đạt ngưỡng. Trong
Release, quyền SSH chỉ được mở cho đúng IP runner trong thời gian job deploy.

## 2.7 Chạy và xác nhận

Push code và file `.dvc` lên nhánh `main`, sau đó theo dõi tab **Actions**. Có thể dùng
**workflow_dispatch** để chạy thử thủ công. Tất cả bốn jobs phải xanh.

Từ máy cá nhân gọi API bằng địa chỉ EC2:

```bash
curl http://<EC2_PUBLIC_IP>:8080/healthz
curl -X POST http://<EC2_PUBLIC_IP>:8080/score \
  -H "Content-Type: application/json" \
  -d '{"features": [28, 2, 14, 2, 11, 0, 1, 0, 0, 45]}'
```

Kết quả health là `{"status":"ok"}`. Score trả `prediction` 0/1 và nhãn
`thu_nhap_thap` hoặc `thu_nhap_cao`.

Nếu service lỗi, SSH vào EC2 và xem:

```bash
sudo systemctl status income-api --no-pager
sudo journalctl -u income-api -n 50 --no-pager
```

Chụp ảnh theo [README ảnh nộp bài](../nop-bai/anh-chup-man-hinh/README.md). Trước khi
kết thúc lab, xóa EC2 và các tài nguyên AWS không dùng nữa; kiểm tra Billing. Tắt EC2
không nhất thiết xóa mọi tài nguyên có thể phát sinh chi phí.

---

Tiếp theo: [Bước 3 - Huấn luyện liên tục](buoc-3.md)
