# AI Stock 文件索引

這個目錄是本專案的程式碼導覽。文件以「責任邊界、入口、資料流、持久化位置、已知限制」為主，方便人與 AI 在修改程式前快速建立正確上下文。

## 建議閱讀順序

1. [系統總覽](00-system-overview.md)：先理解分層與主要資料流。
2. [應用程式入口](01-application-entry.md)：了解 Streamlit 啟動、Session State 與每日排程。
3. [資料服務](02-data-services.md)：了解股票、財務、新聞與市場情緒資料從哪裡來。
4. [AI 與策略](03-ai-and-strategy.md)：了解 AIAdvisor、AIService、技術指標與策略評分。
5. [交易核心](04-trading-core.md)：了解 PaperAccount、AutoTrader 與 AutoTraderV2。
6. [排程任務](05-jobs-and-scheduling.md)：了解早盤預測、收盤驗證、掃描與自動交易。
7. [Streamlit UI](06-streamlit-ui.md)：了解頁面入口與 UI 子模組的資料依賴。
8. [快取與持久化](07-cache-and-persistence.md)：了解 JSON、CSV、SQLite、Parquet 與模型檔案。
9. [開發與維護](08-development-guide.md)：了解本機啟動、測試、環境變數與修改注意事項。

## 快速定位

- 使用者進入點：`app.py`
- 顧問外觀層：`ai_advisor.py`
- 交易帳戶：`core/trader.py`
- 舊版自動交易：`core/auto_trader.py`
- V2 自動交易：`core/auto_trader_v2.py`
- AI 與模型服務：`services/ai_service.py`
- 技術指標：`services/tech_analysis.py`
- 交易評分策略：`services/stock_strategy.py`
- 市場資料：`services/market_data.py`、`utils.py`
- 每日任務：`jobs/daily_job.py`
- UI 組裝：`pages_ui/`

## 重要現況

- 本系統是模擬交易平台，不會直接連接券商下單。
- `AutoTraderV2` 是主要的多訊號自動交易實作；`AutoTrader` 是較早的單一標的版本。
- UI 多處顯示「LSTM 已移除」，但 `AIService` 與模型定義仍保留 LSTM/HybridTransLSTM 相容程式。修改 AI 流程前，請以實際呼叫路徑為準，不要只依 UI 文案判斷。
- 多數資料路徑是相對於目前工作目錄的 `data/`。從錯誤目錄啟動會造成快取、帳戶與模型找不到。
