# 應用程式入口

## 入口檔案

`app.py` 是 Streamlit 的唯一主入口，啟動方式是：

```text
streamlit run app.py
```

## 啟動順序

1. 匯入 `PaperAccount`、`AIAdvisor`、兩個自動交易器、每日任務與 UI renderer。
2. 呼叫 `st.set_page_config()`。
3. 透過 `@st.cache_resource` 啟動一個背景排程執行緒，避免 Streamlit rerun 重複建立。
4. 初始化 `st.session_state` 中的帳戶、顧問、交易器與購物清單。
5. 呼叫 `render_sidebar()` 取得目前股票、歷史資料、現價與股票清單。
6. 建立五個分頁：K 線分析、財報分析、AI 投顧、自動交易、AI 實驗室。

## Session State

- `account`：使用者模擬帳戶，預設 `data/account_data.json`。
- `ai_advisor`：目前頁面的 `AIAdvisor` 實例。
- `auto_trader`：V1 AI 帳戶，預設 `data/ai_account.json`。
- `auto_trader_v2`：V2 AI 帳戶，預設 `data/ai_account_v2.json`。
- `ai_shopping_list`：頁面使用的 DataFrame。

Streamlit 重新執行腳本時會重建區域變數，但保留 Session State。`app.py` 也會檢查舊的 `ai_advisor` 實例是否缺少 `force` 參數，以便類別更新後替換實例。

## 背景排程

排程每 30 秒檢查一次時間：

- 平日 08:30：建立獨立 `AIAdvisor`，執行 `run_morning_predictions()`。
- 平日 14:30：執行 `run_daily_job()`，包含收盤驗證、掃描與自動交易。

這是程序內排程，不是 Windows 工作排程器本身。若 Streamlit 程序停止，背景執行緒也會停止。

## 風險與注意事項

- `st.cache_resource` 只保證同一個 Streamlit 程序內不重複建立資源；多程序部署仍可能各自執行一次排程。
- 背景任務使用獨立顧問，不能依賴 `st.session_state` 或 UI context。
- 相對路徑均假設從專案根目錄啟動。
