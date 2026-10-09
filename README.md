# Sentiment-Aware Recommendation (Olist Dataset)

Ứng dụng sentiment analysis từ nội dung review để tinh chỉnh điểm đánh giá
(`review_score`) thành điểm điều chỉnh (`adjusted_score`), sau đó so sánh
2 mô hình gợi ý ALS: một train trên điểm gốc, một train trên điểm đã điều
chỉnh, nhằm xem sentiment có giúp cải thiện chất lượng gợi ý hay không.

## Cấu trúc thư mục

```
Sentiment/
├── data/
│   ├── raw/                             # CSV gốc của bộ dữ liệu Olist
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   └── product_category_name_translation.csv
│   └── processed/
│       └── reviews_sentiment.parquet    # review + sentiment_score/sentiment_stars/is_fake
├── models/
│   ├── model_baseline/                  # Spark ALSModel train trên review_score gốc
│   │   ├── userFactors/ itemFactors/ metadata/
│   └── model_sentiment/                 # Spark ALSModel train trên adjusted_score (ALPHA tối ưu)
│       ├── userFactors/ itemFactors/ metadata/
├── src/
│   ├── config.py                        # cấu hình dùng chung (RAW_DIR, PROCESSED_DIR, MODELS_DIR, Mongo URI)
│   ├── 39_load_to_mongo.ipynb           # Bước 1: nạp CSV (data/raw) vào MongoDB
│   ├── 39_sentiment_analysis.ipynb      # Bước 2: chấm sentiment -> data/processed
│   └── 39_als_train.ipynb               # Bước 3: lọc dữ liệu, grid-search ALPHA, train & lưu model vào models/
├── requirements.txt
└── README.md
```

## Pipeline

1. **`39_load_to_mongo.ipynb`** — Đọc 5 file CSV Olist bằng Spark, nạp vào
   MongoDB (database `olist`). Ghi theo kiểu idempotent (ghi collection tạm
   rồi `rename`) để không mất dữ liệu nếu lỗi giữa chừng.

2. **`39_sentiment_analysis.ipynb`**
   - Lấy các review có nội dung bình luận từ MongoDB.
   - Chấm sentiment **trực tiếp trên tiếng Bồ Đào Nha gốc** bằng model đa
     ngôn ngữ, đa lớp `nlptown/bert-base-multilingual-uncased-sentiment`
     (output 1–5 sao) — **không qua bước dịch**.
   - Chuẩn hoá số sao thành `sentiment_score` (0–1).
   - Gắn cờ `is_fake` (heuristic: `|sentiment_stars - review_score| > 2`),
     kèm bước lấy mẫu ngẫu nhiên để tự kiểm tra thủ công độ tin cậy của
     ngưỡng này.
   - Lưu `reviews_sentiment.parquet` và collection MongoDB `reviews_sentiment`
     (ghi idempotent).
   - **Không tính sẵn `adjusted_score`** — việc này chuyển sang bước 3 vì
     `ALPHA` cần được chọn bằng grid-search, không cố định trước.

3. **`39_als_train.ipynb`**
   - Lọc chỉ giữ đơn hàng có đúng 1 sản phẩm (`order_item_id` duy nhất),
     tránh 1 review bị gán nhầm cho nhiều sản phẩm khác nhau trong cùng đơn.
   - Ghép `order_items + orders + reviews_sentiment`, encode
     `customer_id`/`product_id` bằng `StringIndexer`.
   - **Split train/test 1 lần duy nhất**, dùng chung cho mọi thí nghiệm
     (tránh lệch dòng do split nhiều lần độc lập).
   - Định nghĩa "item relevant" để đánh giá ranking dựa trên `review_score`
     **gốc** (`>= 4`) — độc lập với `adjusted_score`, tránh đánh giá circular.
   - Train model baseline (label = `review_score`).
   - **Grid-search `ALPHA`** trong `adjusted_score = ALPHA*sentiment_score*5
     + (1-ALPHA)*review_score` (thử 0.1 → 0.9), chọn theo NDCG@10.
   - Train model sentiment với `ALPHA` tốt nhất.
   - Đánh giá bằng Precision@10/NDCG@10 qua `pyspark.mllib.evaluation.RankingMetrics`
     (built-in, không tự viết UDF).
   - **RMSE chỉ để model tự đánh giá với chính target của nó** — không so
     sánh % chênh lệch RMSE giữa 2 model vì 2 target khác thang đo.
   - Lưu cả 2 model bằng `.write().overwrite().save(...)`.

## Cách chạy

```bash
pip install -r requirements.txt
```

- Cài JDK 17, set biến môi trường `JAVA_HOME`.
- Khởi động MongoDB (mặc định `mongodb://localhost:27017/`, đổi qua biến môi
  trường `MONGO_URI` nếu cần).
- (Tuỳ chọn) set `SENTIMENT_DATA_DIR`/`SENTIMENT_RAW_DIR`/`SENTIMENT_PROCESSED_DIR`/
  `SENTIMENT_MODELS_DIR` nếu muốn đặt `data/raw`, `data/processed`, `models/`
  ở nơi khác thay vì mặc định theo cấu trúc `Sentiment/data/...` và
  `Sentiment/models/...` ở trên.
- Chạy tuần tự: `39_load_to_mongo.ipynb` → `39_sentiment_analysis.ipynb` →
  `39_als_train.ipynb`.

## Đã sửa so với bản gốc

| # | Vấn đề cũ | Cách sửa |
|---|---|---|
| 1 | Hard-code path Windows cá nhân | `config.py` đọc từ biến môi trường, fallback hợp lý |
| 2 | 1 review nhân bản cho nhiều sản phẩm trong cùng đơn | Lọc chỉ giữ đơn 1 sản phẩm trước khi ghép |
| 3 | Dịch PT→EN bằng ghép/tách chuỗi `\|\|\|`, rủi ro lệch dòng | Bỏ dịch, dùng model sentiment đa ngôn ngữ chạy trực tiếp trên PT |
| 4 | Sentiment nhị phân, train trên miền phim ảnh (SST-2) | Model đa lớp 1–5 sao, đa ngôn ngữ, cùng thang với `review_score` |
| 5 | `ALPHA=0.4` chọn tuỳ ý | Grid-search 0.1–0.9 theo NDCG@10 |
| 6 | `is_fake` chưa validate | Thêm bước lấy mẫu để tự kiểm tra thủ công |
| 7 | So sánh RMSE giữa 2 target khác thang đo | Bỏ so sánh chéo, dùng Precision@10/NDCG@10 làm căn cứ chính |
| 8 | Đánh giá "relevant" dựa trên `adjusted_score` → circular | Đổi sang dùng `review_score` gốc làm ground-truth độc lập |
| 9 | Tự viết UDF tính Precision@10 | Dùng `pyspark.mllib.evaluation.RankingMetrics` (built-in) |
| 10 | Thiếu code lưu model | Thêm `.write().overwrite().save(...)` |
| 11 | `MongoDB drop()` rồi `insert_many()`, mất dữ liệu nếu lỗi giữa chừng | Ghi vào collection tạm rồi `rename` (idempotent) |
| 12 | Batch translate/sentiment lỗi mất cả batch | Retry từng câu lẻ khi batch lỗi |

## Hướng nâng cấp tiếp theo (nếu viết báo cáo/bài báo)

- **McAuley & Leskovec (2013), "Hidden Factors as Topics", RecSys** — cơ sở
  lý luận kết hợp review text vào rating thay vì công thức cộng tuyến tính.
- **Ott et al. (2011), "Finding Deceptive Opinion Spam...", ACL** — phương
  pháp phát hiện review giả bài bản hơn ngưỡng heuristic `is_fake`.
- **Cremonesi, Koren & Turrin (2010), "Performance of Recommender Algorithms
  on Top-N Recommendation Tasks", RecSys** — lý do nên ưu tiên Precision@N/NDCG
  thay vì so RMSE trực tiếp giữa các mô hình.
