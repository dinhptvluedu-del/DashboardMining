"""
XGBoost ML Studio - Ứng dụng Phân tích dữ liệu & Huấn luyện mô hình XGBoost
Thiết kế cho tập dữ liệu: Naive-Bayes-Classification-Data.csv
Tác giả: Dinh PT (dinh.pt.vlu.edu@gmail.com)
Bản quyền © 2026 - Bảo lưu mọi quyền.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime

# Machine Learning & Metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. Cấu hình trang & Giao diện (Page Config & Styling)
# -------------------------------------------------------------
st.set_page_config(
    page_title="XGBoost ML Studio | Naive-Bayes Data",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern clean aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Quản lý Session State (Session State Management)
# -------------------------------------------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_params' not in st.session_state:
    st.session_state.model_params = None
if 'eval_metrics' not in st.session_state:
    st.session_state.eval_metrics = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'history_logs' not in st.session_state:
    st.session_state.history_logs = []

# -------------------------------------------------------------
# 2. Tải dữ liệu (Data Loading)
# -------------------------------------------------------------
@st.cache_data
def load_default_data():
    csv_paths = [
        "Naive-Bayes-Classification-Data.csv",
        "public/Naive-Bayes-Classification-Data.csv",
        "../Naive-Bayes-Classification-Data.csv"
    ]
    for p in csv_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    # Synthetic fallback if not found
    np.random.seed(42)
    n = 995
    glucose = np.random.normal(44.3, 10.5, n).clip(20, 70).round(1)
    bp = np.random.normal(79.2, 11.2, n).clip(50, 100).round(1)
    # Decision boundary roughly glucose > 45 or bp < 75
    prob = 1 / (1 + np.exp(-(0.12 * (glucose - 44) - 0.08 * (bp - 79))))
    diabetes = (np.random.rand(n) < prob).astype(int)
    return pd.DataFrame({"glucose": glucose, "bloodpressure": bp, "diabetes": diabetes})

# -------------------------------------------------------------
# 3. Sidebar Navigation & Data Selection
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("XGBoost Studio")
    st.caption("Phiên bản 2.5 - Hỗ trợ phân tích dữ liệu Naive-Bayes & Phân loại học máy")
    st.divider()

    st.subheader("📁 Nguồn dữ liệu")
    data_source = st.radio(
        "Chọn nguồn dữ liệu:",
        ["Mặc định (Naive-Bayes-Classification-Data.csv)", "Tải lên tệp CSV tùy chỉnh"],
        index=0
    )

    df = None
    if data_source == "Mặc định (Naive-Bayes-Classification-Data.csv)":
        df = load_default_data()
        st.success(" Đã tải tệp Naive-Bayes-Classification-Data.csv thành công!")
    else:
        uploaded_file = st.file_uploader("Chọn tệp CSV từ máy tính", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f" Đã tải tệp: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                df = load_default_data()
        else:
            st.info("Vui lòng tải lên tệp CSV hoặc dùng dữ liệu mặc định.")
            df = load_default_data()

    st.divider()
    st.subheader("🧭 Điều hướng tính năng")
    menu_choice = st.radio(
        "Chuyển đến phần:",
        [
            "1. Dashboard Tổng kết",
            "2. Thống kê mô tả & Tương quan",
            "3. Tiền xử lý dữ liệu",
            "4. Huấn luyện XGBoost",
            "5. Đánh giá mô hình",
            "6. Dự đoán dữ liệu mới",
            "7. Lịch sử thử nghiệm"
        ],
        index=0
    )

    st.divider()
    st.markdown("""
    **Thông tin hệ thống**
    - Tập dữ liệu: `{rows}` dòng × `{cols}` cột
    - Trạng thái mô hình: `{status}`
    """.format(
        rows=len(df),
        cols=len(df.columns),
        status="Đã huấn luyện" if st.session_state.model is not None else "Chưa huấn luyện"
    ))

# -------------------------------------------------------------
# HEADER CHÍNH
# -------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Hệ Thống Phân Tích & Huấn Luyện XGBoost</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân tích toàn diện, tiền xử lý và xây dựng mô hình Extreme Gradient Boosting cho bài toán phân loại</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 1: DASHBOARD TỔNG KẾT
# -------------------------------------------------------------
if menu_choice == "1. Dashboard Tổng kết":
    st.header("📊 1. Dashboard Tổng Kết Dữ Liệu")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số dòng (Mẫu)", f"{len(df):,}")
    with col2:
        st.metric("Tổng số cột (Đặc trưng)", len(df.columns))
    with col3:
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Số cột dạng số", num_cols)
    with col4:
        missing_total = df.isnull().sum().sum()
        st.metric("Giá trị khuyết thiếu (NaN)", missing_total)

    st.subheader("📋 Bảng xem trước dữ liệu (First 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("🔍 Kiểu dữ liệu và thông tin chi tiết")
    info_df = pd.DataFrame({
        "Tên cột": df.columns,
        "Kiểu dữ liệu": df.dtypes.astype(str),
        "Số giá trị duy nhất (Unique)": [df[c].nunique() for c in df.columns],
        "Số giá trị khuyết (NaN)": [df[c].isnull().sum() for c in df.columns],
        "Tỷ lệ khuyết (%)": [round(df[c].isnull().mean() * 100, 2) for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("📈 Trực quan hóa phân phối biến số")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) >= 2:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            target_candidate = "diabetes" if "diabetes" in df.columns else df.columns[-1]
            fig_target = px.histogram(
                df, x=target_candidate, color=target_candidate,
                title=f"Phân phối biến mục tiêu ({target_candidate})",
                color_discrete_sequence=["#3b82f6", "#ef4444"]
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col_chart2:
            feat_x = "glucose" if "glucose" in df.columns else numeric_columns[0]
            feat_y = "bloodpressure" if "bloodpressure" in df.columns else (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])
            fig_scatter = px.scatter(
                df, x=feat_x, y=feat_y, color=target_candidate,
                title=f"Biểu đồ phân tán: {feat_x} vs {feat_y}",
                color_discrete_sequence=["#3b82f6", "#ef4444"],
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: THỐNG KÊ MÔ TẢ
# -------------------------------------------------------------
elif menu_choice == "2. Thống kê mô tả & Tương quan":
    st.header("📐 2. Thống Kê Mô Tả & Tương Quan Biến Số")

    st.subheader("📋 Bảng thống kê mô tả chi tiết (Descriptive Statistics)")
    desc_df = df.describe().T
    desc_df["median"] = df.median(numeric_only=True)
    desc_df["skewness"] = df.skew(numeric_only=True)
    # Format columns nicely
    st.dataframe(desc_df[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skewness"]].round(3), use_container_width=True)

    st.divider()
    col_corr, col_dist = st.columns([1, 1])

    with col_corr:
        st.subheader("🔥 Ma trận tương quan (Correlation Heatmap)")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_mat = numeric_df.corr()
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Ma trận hệ số tương quan Pearson"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Cần ít nhất 2 biến số để tính ma trận tương quan.")

    with col_dist:
        st.subheader("📊 Phân phối các biến đặc trưng (KDE & Histogram)")
        selected_num_col = st.selectbox("Chọn biến số để xem phân phối:", numeric_df.columns)
        fig_hist = px.histogram(
            df, x=selected_num_col, marginal="box",
            nbins=30, title=f"Phân phối tần suất của cột '{selected_num_col}'",
            color_discrete_sequence=["#10b981"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: TIỀN XỬ LÝ DỮ LIỆU
# -------------------------------------------------------------
elif menu_choice == "3. Tiền xử lý dữ liệu":
    st.header("⚙️ 3. Tiền Xử Lý Dữ Liệu Đầu Vào")

    st.markdown("Cấu hình chuẩn bị dữ liệu trước khi đưa vào mô hình học máy XGBoost:")

    col_target, col_features = st.columns([1, 2])
    with col_target:
        default_target_idx = df.columns.get_loc("diabetes") if "diabetes" in df.columns else len(df.columns) - 1
        target_column = st.selectbox(
            "1. Chọn biến mục tiêu (Target column):",
            df.columns,
            index=default_target_idx
        )
        st.session_state.target_col = target_column

    with col_features:
        candidate_features = [c for c in df.columns if c != target_column]
        feature_columns = st.multiselect(
            "2. Chọn các biến đầu vào (Feature columns):",
            candidate_features,
            default=candidate_features
        )
        st.session_state.feature_cols = feature_columns

    if not feature_columns:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 biến đầu vào.")
    else:
        st.divider()
        col_prep1, col_prep2, col_prep3 = st.columns(3)

        with col_prep1:
            st.subheader("🧹 Xử lý giá trị khuyết")
            impute_opt = st.selectbox(
                "Phương pháp xử lý missing values:",
                ["Điền giá trị Trung bình (Mean)", "Điền giá trị Trung vị (Median)", "Xóa các dòng chứa NaN (Drop)"]
            )

        with col_prep2:
            st.subheader("📏 Chuẩn hóa dữ liệu số")
            scaler_opt = st.selectbox(
                "Phương pháp chuẩn hóa (Scaling):",
                ["StandardScaler (Z-score: Mean=0, Std=1)", "MinMaxScaler (Thang đo 0 - 1)", "Không chuẩn hóa (None)"]
            )

        with col_prep3:
            st.subheader("✂️ Phân chia Train / Test")
            test_size = st.slider("Tỷ lệ tập kiểm thử (Test Size):", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
            random_seed = st.number_input("Random Seed:", value=42, min_value=1, max_value=9999)

        if st.button("🚀 Thực hiện Tiền Xử Lý & Lưu Cấu Hình", type="primary"):
            with st.spinner("Đang xử lý dữ liệu..."):
                # Clean dataframe
                sub_df = df[[target_column] + feature_columns].copy()

                if impute_opt == "Xóa các dòng chứa NaN (Drop)":
                    sub_df = sub_df.dropna()
                else:
                    for col in feature_columns:
                        if pd.api.types.is_numeric_dtype(sub_df[col]):
                            if impute_opt == "Điền giá trị Trung vị (Median)":
                                sub_df[col] = sub_df[col].fillna(sub_df[col].median())
                            else:
                                sub_df[col] = sub_df[col].fillna(sub_df[col].mean())
                        else:
                            sub_df[col] = sub_df[col].fillna(sub_df[col].mode()[0] if not sub_df[col].mode().empty else "Unknown")

                X = sub_df[feature_columns].copy()
                y = sub_df[target_column].copy()

                # Encode categorical features if any
                for col in X.columns:
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))

                # Encode target
                if not pd.api.types.is_numeric_dtype(y):
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y.astype(str))
                else:
                    y = y.astype(int)

                # Train test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed, stratify=y if len(np.unique(y)) == 2 else None
                )

                # Scaling
                scaler = None
                if scaler_opt.startswith("StandardScaler"):
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                elif scaler_opt.startswith("MinMaxScaler"):
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                else:
                    X_train = X_train.values
                    X_test = X_test.values

                st.session_state.scaler = scaler
                st.session_state.preprocessed_data = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train.values if hasattr(y_train, 'values') else y_train,
                    "y_test": y_test.values if hasattr(y_test, 'values') else y_test,
                    "feature_names": feature_columns,
                    "target_name": target_column,
                    "classes": np.unique(y)
                }
                st.success(f"✅ Đã tiền xử lý thành công! Tập Train: {len(X_train)} dòng | Tập Test: {len(X_test)} dòng")

# -------------------------------------------------------------
# TAB 4: HUẤN LUYỆN MÔ HÌNH XGBOOST
# -------------------------------------------------------------
elif menu_choice == "4. Huấn luyện XGBoost":
    st.header("⚡ 4. Cấu Hình & Huấn Luyện Mô Hình XGBoost")

    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ Bạn chưa thực hiện tiền xử lý dữ liệu. Vui lòng vào mục '3. Tiền xử lý dữ liệu' trước!")
    else:
        pdata = st.session_state.preprocessed_data
        st.info(f"Dữ liệu sẵn sàng: {len(pdata['X_train'])} mẫu huấn luyện, {len(pdata['feature_names'])} đặc trưng ({', '.join(pdata['feature_names'])}).")

        st.subheader("🎛️ Thiết lập Siêu tham số (Hyperparameters)")
        col_hp1, col_hp2, col_hp3 = st.columns(3)

        with col_hp1:
            n_estimators = st.slider("Số lượng cây (n_estimators):", 10, 300, 60, step=10)
            learning_rate = st.slider("Tốc độ học (learning_rate / eta):", 0.01, 0.5, 0.1, step=0.01)

        with col_hp2:
            max_depth = st.slider("Độ sâu tối đa của cây (max_depth):", 1, 10, 4)
            subsample = st.slider("Tỷ lệ chọn mẫu hàng (subsample):", 0.5, 1.0, 0.8, step=0.05)

        with col_hp3:
            colsample_bytree = st.slider("Tỷ lệ chọn cột đặc trưng (colsample_bytree):", 0.5, 1.0, 1.0, step=0.05)
            gamma = st.slider("Hệ số phạt nút lá (gamma / min_split_loss):", 0.0, 5.0, 0.0, step=0.1)

        st.divider()

        if st.button("🚀 Huấn Luyện Mô Hình XGBoost (Train Model)", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()
            status_text.text("Đang khởi tạo thuật toán XGBoost...")
            progress_bar.progress(20)
            time.sleep(0.2)

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                eval_metric="logloss"
            )

            status_text.text("Đang khớp mô hình trên tập Train...")
            progress_bar.progress(60)

            model.fit(
                pdata["X_train"],
                pdata["y_train"],
                eval_set=[(pdata["X_train"], pdata["y_train"]), (pdata["X_test"], pdata["y_test"])],
                verbose=False
            )

            progress_bar.progress(90)
            status_text.text("Đang tính toán chỉ số đánh giá...")

            # Predictions
            y_pred = model.predict(pdata["X_test"])
            y_prob = model.predict_proba(pdata["X_test"])[:, 1]

            acc = accuracy_score(pdata["y_test"], y_pred)
            prec = precision_score(pdata["y_test"], y_pred, zero_division=0)
            rec = recall_score(pdata["y_test"], y_pred, zero_division=0)
            f1 = f1_score(pdata["y_test"], y_pred, zero_division=0)
            auc = roc_auc_score(pdata["y_test"], y_prob)

            elapsed = round(time.time() - start_time, 2)
            progress_bar.progress(100)
            status_text.text("Huấn luyện hoàn tất!")

            # Save in session state
            st.session_state.model = model
            st.session_state.model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "gamma": gamma
            }
            st.session_state.eval_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "elapsed": elapsed
            }

            # Save to history
            st.session_state.history_logs.append({
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Số cây (n_est)": n_estimators,
                "Learning Rate": learning_rate,
                "Max Depth": max_depth,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
                "Thời gian chạy (s)": elapsed
            })

            st.success(f"🎉 Huấn luyện thành công trong {elapsed} giây! Độ chính xác (Accuracy): {acc * 100:.2f}% | ROC-AUC: {auc:.4f}")
            st.info("👉 Hãy chuyển sang tab '5. Đánh giá mô hình' để xem chi tiết ma trận nhầm lẫn và đường cong ROC!")

# -------------------------------------------------------------
# TAB 5: ĐÁNH GIÁ MÔ HÌNH
# -------------------------------------------------------------
elif menu_choice == "5. Đánh giá mô hình":
    st.header("📈 5. Đánh Giá Kết Quả Phân Loại")

    if st.session_state.model is None or st.session_state.eval_metrics is None:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng chuyển sang tab '4. Huấn luyện XGBoost' để train mô hình!")
    else:
        m = st.session_state.eval_metrics
        pdata = st.session_state.preprocessed_data
        model = st.session_state.model

        st.subheader("🎯 Các chỉ số đánh giá cốt lõi trên tập Test")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
        c2.metric("Precision", f"{m['precision'] * 100:.2f}%")
        c3.metric("Recall", f"{m['recall'] * 100:.2f}%")
        c4.metric("F1-Score", f"{m['f1_score'] * 100:.2f}%")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

        st.divider()
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("🧩 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(pdata["y_test"], m["y_pred"])
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Âm tính (0)", "Dương tính (1)"],
                yticklabels=["Âm tính (0)", "Dương tính (1)"],
                ax=ax
            )
            ax.set_xlabel("Dự đoán (Predicted)")
            ax.set_ylabel("Thực tế (Actual)")
            st.pyplot(fig_cm)

        with col_roc:
            st.subheader("📈 Đường cong ROC-AUC")
            fpr, tpr, _ = roc_curve(pdata["y_test"], m["y_prob"])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'XGBoost (AUC = {m["roc_auc"]:.3f})', line=dict(color='#2563eb', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline (Ngẫu nhiên)', line=dict(dash='dash', color='gray')))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC)",
                xaxis_title="False Positive Rate (FPR)",
                yaxis_title="True Positive Rate (TPR)",
                width=450, height=350
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()
        col_fi, col_pr = st.columns(2)

        with col_fi:
            st.subheader("🌟 Mức độ quan trọng của đặc trưng (Feature Importance)")
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Đặc trưng": pdata["feature_names"],
                "Độ quan trọng": importances
            }).sort_values(by="Độ quan trọng", ascending=True)

            fig_fi = px.bar(
                fi_df, x="Độ quan trọng", y="Đặc trưng", orientation="h",
                title="Feature Importance từ XGBoost",
                color="Độ quan trọng", color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_pr:
            st.subheader("📉 Đường cong Precision-Recall")
            precision_vals, recall_vals, _ = precision_recall_curve(pdata["y_test"], m["y_prob"])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall', line=dict(color='#10b981', width=3)))
            fig_pr.update_layout(
                title="Precision vs Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                width=450, height=350
            )
            st.plotly_chart(fig_pr, use_container_width=True)

# -------------------------------------------------------------
# TAB 6: DỰ ĐOÁN DỮ LIỆU BÊN NGOÀI
# -------------------------------------------------------------
elif menu_choice == "6. Dự đoán dữ liệu mới":
    st.header("🔮 6. Dự Đoán / Phân Loại Dữ Liệu Mới")

    if st.session_state.model is None:
        st.warning("⚠️ Bạn cần huấn luyện mô hình trước khi thực hiện dự đoán!")
    else:
        model = st.session_state.model
        scaler = st.session_state.scaler
        pdata = st.session_state.preprocessed_data
        features = pdata["feature_names"]

        predict_mode = st.radio("Chọn phương thức dự đoán:", ["Nhập thủ công từng mẫu", "Tải lên tệp CSV hàng loạt"], horizontal=True)

        if predict_mode == "Nhập thủ công từng mẫu":
            st.subheader("📝 Nhập các giá trị đặc trưng:")
            input_vals = {}
            cols = st.columns(min(len(features), 3))
            for i, f in enumerate(features):
                with cols[i % len(cols)]:
                    input_vals[f] = st.number_input(f"Nhập giá trị '{f}':", value=50.0, step=1.0)

            if st.button("🔮 Dự đoán kết quả", type="primary"):
                input_df = pd.DataFrame([input_vals])
                raw_input = input_df[features].values
                if scaler is not None:
                    scaled_input = scaler.transform(raw_input)
                else:
                    scaled_input = raw_input

                prob = model.predict_proba(scaled_input)[0, 1]
                pred = int(prob >= 0.5)

                st.divider()
                st.subheader("Kết quả dự đoán:")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if pred == 1:
                        st.error(f"🔴 KẾT QUẢ DỰ ĐOÁN: DƯƠNG TÍNH (Lớp 1 - Có nguy cơ)")
                    else:
                        st.success(f"🟢 KẾT QUẢ DỰ ĐOÁN: ÂM TÍNH (Lớp 0 - Bình thường)")
                with res_col2:
                    st.metric("Xác suất Dương tính (Probability)", f"{prob * 100:.2f}%")
                    st.progress(float(prob))

        else:
            st.subheader("📂 Tải lên tệp CSV để dự đoán hàng loạt")
            st.caption(f"Tệp CSV cần có các cột: {', '.join(features)}")
            pred_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"], key="pred_csv")

            if pred_file is not None:
                new_df = pd.read_csv(pred_file)
                st.write("Bản ghi xem trước từ file tải lên:")
                st.dataframe(new_df.head(), use_container_width=True)

                missing_feats = [f for f in features if f not in new_df.columns]
                if missing_feats:
                    st.error(f"Tệp CSV thiếu các cột bắt buộc: {', '.join(missing_feats)}")
                else:
                    if st.button("⚡ Thực hiện dự đoán trên toàn bộ tệp"):
                        pred_input = new_df[features].values
                        if scaler is not None:
                            pred_scaled = scaler.transform(pred_input)
                        else:
                            pred_scaled = pred_input

                        probs = model.predict_proba(pred_scaled)[:, 1]
                        preds = (probs >= 0.5).astype(int)

                        result_df = new_df.copy()
                        result_df["Xac_Suat_Duong_Tinh"] = (probs * 100).round(2)
                        result_df["Du_Doan_Nhan"] = preds
                        result_df["Ket_Luan"] = np.where(preds == 1, "Dương tính", "Âm tính")

                        st.success(f" Đã dự đoán xong {len(result_df)} dòng dữ liệu!")
                        st.dataframe(result_df, use_container_width=True)

                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Tải xuống kết quả dự đoán (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="ket_qua_du_doan_xgboost.csv",
                            mime="text/csv"
                        )

# -------------------------------------------------------------
# TAB 7: LỊCH SỬ THỰC HIỆN
# -------------------------------------------------------------
elif menu_choice == "7. Lịch sử thử nghiệm":
    st.header("📜 7. Lịch Sử Huấn Luyện & Thử Nghiệm Mô Hình")

    if not st.session_state.history_logs:
        st.info("Chưa có lần huấn luyện nào được ghi nhận. Hãy huấn luyện mô hình để theo dõi lịch sử tại đây.")
    else:
        history_df = pd.DataFrame(st.session_state.history_logs)
        st.dataframe(history_df, use_container_width=True)

        st.subheader("📊 So sánh độ chính xác giữa các lần thử nghiệm")
        fig_hist_acc = px.line(
            history_df.reset_index(),
            x="index",
            y=["Accuracy", "F1-Score", "ROC-AUC"],
            markers=True,
            title="Biến thiên các chỉ số đánh giá qua các lần huấn luyện",
            labels={"index": "Lần thử nghiệm (Run #)", "value": "Điểm số"}
        )
        st.plotly_chart(fig_hist_acc, use_container_width=True)

        csv_history = io.StringIO()
        history_df.to_csv(csv_history, index=False)
        st.download_button(
            label="📥 Tải xuống lịch sử thử nghiệm (CSV)",
            data=csv_history.getvalue(),
            file_name="lich_su_huan_luyen_xgboost.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# FOOTER BẢN QUYỀN
# -------------------------------------------------------------
st.markdown("""
<div class="footer">
    <p><strong>© 2026 - Phát triển bởi Dinh PT (VLU). Bảo lưu mọi quyền.</strong></p>
    <p>XGBoost ML Studio • Phân tích dữ liệu & Machine Learning Pipeline • Hỗ trợ tệp Naive-Bayes-Classification-Data.csv</p>
</div>
""", unsafe_allow_html=True)
