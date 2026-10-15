# Streamlit UI

## UI 組裝方式

`app.py` 負責頁面與狀態，`pages_ui/` 負責各畫面內容。UI 函式通常採用 `render_*` 命名，接收已準備好的資料與服務實例。

### 目前主頁分頁

- `ui_analysis.py`：K 線與技術分析。
- `ui_financial.py`：財務報表與比率。
- `ui_advisor.py` / `advisor/`：AI 個股分析、聊天與驗證。
- `ui_battle.py` / `battle/`：市場掃描、自動交易與回測。
- `ui_testing.py`：API 與系統診斷。
- `ui_sidebar.py` / `sidebar/`：股票搜尋、行情摘要與使用者下單。

`ui_asset.py` 與 `asset/` 提供資產概況與 AI 經理元件，但目前不在 `app.py` 的五個主分頁中，修改前先確認是否為保留中的遷移模組。

## 側邊欄資料流

`render_sidebar(account, ai_advisor)`：

1. 取得股票清單。
2. 讓 `sub_search.py` 決定目前代碼。
3. 呼叫 `utils.get_stock_data()` 取得行情。
4. 將帳戶、股票代碼與現價交給 `sub_trade_panel.py`。
5. 回傳 `(symbol, hist_data, current_price, stock_info, df_stocks)` 給主頁。

## AI 分析頁

`advisor/sub_analysis.py` 會呼叫 `AIAdvisor.analyze_stock()` 顯示 action、score 與 reasons；預測價格則透過 `predict_next_price()` 取得。`advisor/sub_chat.py` 會先呼叫 `SearchOptimizer.get_cached_summary()`，再把技術、新聞與 AI 診斷組成 Gemini prompt。

## UI 狀態與效能

- 互動事件會造成整個 Streamlit script rerun，因此重量級服務應使用 cache 或明確按鈕觸發。
- `SearchOptimizer` 已用快取、背景 thread 與 timeout 避免聯網工作長時間阻塞 UI。
- UI 不應直接修改交易規則；應呼叫 `core` 或 `services` 的方法。
- 新增 Session State key 時，需記錄初始化時機與型別，避免 rerun 後出現舊資料格式。
