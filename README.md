# Sentiment-Aware Recommendation (Olist Dataset)

Ứng dụng sentiment analysis từ nội dung review để tinh chỉnh điểm đánh giá
(`review_score`) thành điểm điều chỉnh (`adjusted_score`), sau đó so sánh
2 mô hình gợi ý ALS: một train trên điểm gốc, một train trên điểm đã điều
chỉnh, nhằm xem sentiment có giúp cải thiện chất lượng gợi ý hay không.

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
