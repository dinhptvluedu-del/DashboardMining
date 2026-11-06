# XGBoost ML Studio - Phân Tích & Huấn Luyện Mô Hình Phân Loại

Ứng dụng Web hoàn chỉnh phục vụ phân tích dữ liệu, tiền xử lý, huấn luyện mô hình XGBoost và dự đoán phân loại, được tối ưu cho tập dữ liệu `Naive-Bayes-Classification-Data.csv`.

## 📌 Tính năng chính
1. **Giao diện & Bản quyền (UI/UX & Copyright)**: Giao diện hiện đại, sidebar điều hướng mượt mà, chân trang bản quyền © 2026.
2. **Dashboard Tổng Kết Dữ Liệu**: Số lượng dòng, cột, kiểu dữ liệu, biểu đồ phân bố và biểu đồ phân tán.
3. **Thống Kê Mô Tả (Descriptive Statistics)**: Bảng thống kê Mean, Median, Std, Min, Max, Quantiles, Skewness; ma trận tương quan Pearson dạng Heatmap; kiểm tra Missing Values.
4. **Tiền Xử Lý Dữ Liệu (Data Preprocessing)**: Chọn Target column & Feature columns, xử lý missing values (Mean/Median/Drop), mã hóa biến phân loại, chuẩn hóa StandardScaler/MinMaxScaler, chia Train/Test tỷ lệ tùy chỉnh.
5. **Huấn Luyện Mô Hình XGBoost**: Cấu hình các siêu tham số `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `gamma`, nút Huấn luyện kèm thanh tiến trình.
6. **Đánh Giá Kết Quả Phân Loại**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC curve, Precision-Recall curve, Feature Importance bar chart.
7. **Dự Đoán Dữ Liệu Bên Ngoài**: Nhập liệu thủ công từng mẫu đơn lẻ hoặc tải lên file CSV hàng loạt, tải xuống file CSV kết quả dự đoán.
8. **Lịch Sử Thực Hiện**: Lưu log các lần huấn luyện mô hình và so sánh hiệu suất qua các lần thử nghiệm.

---

## 🚀 Hướng dẫn Cài đặt & Chạy ứng dụng Python (Streamlit)

### Bước 1: Yêu cầu môi trường
- Python 3.8+ (khuyên dùng Python 3.10 hoặc 3.11)
- Trình quản lý gói `pip`

### Bước 2: Cài đặt các thư viện cần thiết
Mở Terminal / Command Prompt tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 📂 Cấu trúc thư mục
```
├── Naive-Bayes-Classification-Data.csv   # Tệp dữ liệu mẫu
├── app.py                               # Mã nguồn chính Streamlit
├── requirements.txt                     # Danh sách thư viện Python
├── README.md                            # Hướng dẫn sử dụng
├── src/                                 # Ứng dụng Web React trực quan trên port 3000
└── public/                              # Dữ liệu tĩnh
```

**Bản quyền**: © 2026 - Phát triển bởi Dinh PT (dinh.pt.vlu.edu@gmail.com). Bảo lưu mọi quyền.
