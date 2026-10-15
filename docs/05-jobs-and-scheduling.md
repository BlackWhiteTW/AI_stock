# 排程與批次任務

## 任務入口

`jobs/daily_job.py` 提供可被 Streamlit 背景執行緒、命令列或其他排程器呼叫的函式：

- `run_morning_job()`：建立 `AIAdvisor`，執行早盤預測。
- `run_daily_job()`：收盤驗證、建立購物清單、執行交易決策。
- `daily_model_training()`：呼叫 `MarketTrainingSystem` 進行市場模型訓練。

`jobs/ai_tasks.py` 提供單檔股票掃描與單檔交易決策的工作函式。

## `run_daily_job()` 流程

1. 建立或接收 `AutoTrader`/`AutoTraderV2` 與 `AIAdvisor`。
2. 呼叫 `validate_afternoon_results()` 驗證當日預測。
3. 取得台股清單。
4. 保留舊的 `data/ai_shopping_list.csv`，再隨機補足候選股票。
5. 使用最多 4 個 worker 平行執行市場掃描。
6. 將通過分析的股票寫回 `data/ai_shopping_list.csv`。
7. 合併既有持倉與新購物清單，逐檔執行自動交易。

## 併發與外部服務

掃描工作會同時呼叫行情、技術分析與新聞服務。worker 數量不可任意提高，否則可能觸發資料來源 rate limit。任何單檔失敗通常只記錄錯誤並繼續，不應讓整批任務中止。

## 早盤/收盤時間

`app.py` 內建的時間觸發是本機時間、只在平日執行：08:30 與 14:30。它不是交易所時間同步服務；部署到不同時區或睡眠中的 Windows 電腦時，需另外確認時鐘與程序存活。

## 任務修改注意事項

- 任務函式應接受 `log_func`，避免直接依賴 Streamlit UI。
- 背景任務不要使用 Session State。
- 寫入 CSV/JSON 前應確保 `data/` 存在並使用穩定欄位順序。
- 外部資料不完整時應回傳可辨識的失敗結果，而不是讓平行 worker 拋出未處理例外。
