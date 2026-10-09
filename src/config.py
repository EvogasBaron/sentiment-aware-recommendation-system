"""
Cấu hình dùng chung cho cả 3 notebook (load_to_mongo, sentiment_analysis, als_train).
Không hard-code path máy cá nhân — tất cả lấy từ biến môi trường, có fallback hợp lý.

Cấu trúc thư mục project:

    Sentiment/
    ├── data/
    │   ├── raw/          # CSV gốc (olist_*.csv)
    │   └── processed/    # dữ liệu trung gian (reviews_sentiment.parquet, ...)
    ├── models/
    │   ├── model_baseline/
    │   └── model_sentiment/
    └── src/
        ├── config.py
        ├── 39_load_to_mongo.ipynb
        ├── 39_sentiment_analysis.ipynb
        └── 39_als_train.ipynb

Cách dùng:
    from config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, MONGO_URI, MONGO_DB

Thiết lập (tuỳ chọn) trước khi chạy notebook, ví dụ trong PowerShell:
    $env:SENTIMENT_DATA_DIR = "D:\Downloads\Eco\data"
    $env:SENTIMENT_MODELS_DIR = "D:\Downloads\Eco\models"
    $env:MONGO_URI = "mongodb://localhost:27017/"
    $env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"

Nếu không set, mặc định dùng đúng cấu trúc repo hiện tại:
    <project_root>/data/raw, <project_root>/data/processed, <project_root>/models
"""
import os
import sys
from pathlib import Path

# config.py nằm ở Sentiment/src/config.py -> project root là thư mục cha
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent

# QUAN TRỌNG (Windows): PySpark worker mặc định gọi lệnh "python3" để khởi
# động executor process. Windows không có "python3.exe" (chỉ có "python.exe"),
# nên các job Spark chạy Python code trên executor (ví dụ .rdd.map(...) trong
# RankingMetrics) sẽ lỗi "Cannot run program \'python3\'". Set PYSPARK_PYTHON
# trỏ đúng vào sys.executable (chính Python đang chạy notebook này) để Spark
# dùng đúng interpreter, tự động đúng trên mọi máy/mọi OS, không cần sửa tay.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

DATA_DIR = Path(os.getenv("SENTIMENT_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
RAW_DIR = Path(os.getenv("SENTIMENT_RAW_DIR", str(DATA_DIR / "raw"))).resolve()
PROCESSED_DIR = Path(os.getenv("SENTIMENT_PROCESSED_DIR", str(DATA_DIR / "processed"))).resolve()
MODELS_DIR = Path(os.getenv("SENTIMENT_MODELS_DIR", str(PROJECT_ROOT / "models"))).resolve()

# Đảm bảo các thư mục ghi (processed, models) tồn tại sẵn — tránh lỗi
# "path not found" khi notebook ghi file/model lần đầu.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "olist")

# JAVA_HOME / PYSPARK_PYTHON: KHÔNG set cứng ở đây.
# Set qua biến môi trường hệ thống, hoặc set trong file .env và load bằng
# python-dotenv nếu muốn. Nếu JAVA_HOME chưa có, cảnh báo ngay khi import
# để tránh lỗi khó hiểu từ PySpark về sau.
if not os.getenv("JAVA_HOME"):
    print(
        "[config] CẢNH BÁO: biến môi trường JAVA_HOME chưa được set. "
        "PySpark cần JDK 17. Set JAVA_HOME trước khi tạo SparkSession."
    )


def get_raw_path(filename: str) -> Path:
    """Trả về Path tuyệt đối tới 1 file CSV trong data/raw/."""
    return RAW_DIR / filename


def get_processed_path(filename: str) -> Path:
    """Trả về Path tuyệt đối tới 1 file trong data/processed/."""
    return PROCESSED_DIR / filename


def get_model_path(model_name: str) -> Path:
    """Trả về Path tuyệt đối tới 1 thư mục model trong models/."""
    return MODELS_DIR / model_name