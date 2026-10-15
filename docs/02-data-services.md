# 資料服務

## 入口與責任

### `utils.py`

提供 UI 與任務常用的簡化入口：

- `get_tw_stock_list()`：取得台股/ETF 清單，優先讀取當日 `data/tw_stock_list.csv`，否則從證交所下載。
- `get_stock_data(code)`：將純代碼轉成 `.TW`/`.TWO` 候選，透過 `MarketDataManager` 取得歷史行情。
- `get_stock_with_industry_news(code)`：在行情上加上產業辨識與產業新聞情緒。

`get_tw_stock_list()` 使用 Streamlit data cache，清單至少需要 `Ticker`、`名稱`、`產業` 等欄位。

### `services/market_data.py`

負責行情資料的載入、更新與本地快取。上層應優先呼叫 `utils.get_stock_data()` 或 `MarketDataManager.load_data()`，不要在 UI 中直接散落 Yahoo Finance 呼叫。

### `services/financial_data.py`、`services/finmind_data.py`、`services/mops_data.py`

提供營收、財務比率、財報與台灣市場資料。`tech_analysis.add_financial_features()` 會把財務特徵附加到技術資料中；取得失敗時會使用中性預設值。

### `services/news_analyzer.py`

以關鍵詞權重分析新聞情緒，並負責新聞快取、產業辨識及產業情緒。新聞快取位於 `data/news_cache/`，預設有效 30 分鐘。

### `services/search_engine.py`、`services/search_optimizer.py`

`SearchOptimizer.get_smart_summary()` 是完整智慧摘要流程：行情 -> 技術/財務 -> 聯網新聞 -> 新聞情緒 -> Gemini -> `analysis_report` 快取。聯網新聞最多等待 8 秒，LLM 最多等待 30 秒。

## 外部依賴

主要資料來源與依賴包括 Yahoo Finance、證交所網頁、FinMind/MOPS、新聞網站與可選的 Gemini API。所有外部服務都可能回傳空資料、逾時或格式變動，因此呼叫端應檢查 DataFrame 是否為空。

## 資料品質注意事項

- 技術分析至少建議 60 筆行情，策略會對不足資料返回「未知」。
- 行情欄位名稱使用英文 OHLCV；UI 顯示名稱則可能是中文。
- 股票代碼通常使用 `2330.TW` 形式，檔案名稱會把句點替換成底線的情況需特別注意。
- `force_offline=True` 會直接返回空資料，不會讀取現有快取。
