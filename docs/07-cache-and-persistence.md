# 快取與持久化

## 主要資料位置

| 路徑                          | 用途                          | 主要寫入者                  |
| ----------------------------- | ----------------------------- | --------------------------- |
| `data/tw_stock_list.csv`      | 台股與 ETF 清單               | `utils.get_tw_stock_list()` |
| `data/account_data.json`      | 使用者模擬帳戶                | `PaperAccount`              |
| `data/ai_account.json`        | V1 AI 帳戶                    | `AutoTrader`                |
| `data/ai_account_v2.json`     | app 使用的 V2 AI 帳戶         | `app.py` / `AutoTraderV2`   |
| `data/ai_shopping_list.csv`   | 每日掃描結果                  | `jobs.daily_job`            |
| `data/daily_predictions.json` | 每日預測與驗證                | `AIService`                 |
| `data/saved_models/`          | PyTorch 模型檔                | `AIService` / trainer       |
| `data/training_records/`      | 訓練 loss 與版本紀錄          | `AIService`                 |
| `data/news_cache/`            | 新聞 JSON 快取                | `NewsAnalyzer`              |
| `data/unified_cache/`         | SQLite/Parquet/記憶體統一快取 | `CacheManager`              |

## `CacheManager`

`services/cache_manager.py` 提供三層快取：

1. process memory 的 LRU。
2. SQLite `cache_index.db` 儲存小型 JSON/pickle 與索引。
3. Parquet 儲存大型 pandas DataFrame。

主要 API 是 `set(category, identifier, value, ttl=...)` 與 `get(category, identifier, default=...)`。DataFrame 會使用 Parquet，字典/列表通常使用 JSON。

## 一致性規則

- 快取不是資料真相；資料來源錯誤或格式升級時應允許重新抓取。
- 改變 DataFrame 欄位或序列化格式時，需考慮舊快取能否讀取。
- 帳戶 JSON 是可變狀態，交易成功後才應寫回；測試請使用隔離檔案。
- 模型檔案的 `input_dim`、`hidden_dim`、`num_layers`、`model_type` 需與載入程式相容。
- 相對路徑以程序目前工作目錄為基準，部署腳本應從專案根目錄啟動。

## 清理

專案已有 `scripts/cleanup_cache.py`、`scripts/cache_status.py` 等維護工具。清理前先確認是否正在執行訓練、排程或 Streamlit，避免刪除仍被使用的模型、帳戶或快取檔案。
