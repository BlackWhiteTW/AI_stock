# 開發與維護

## 環境

依賴列在根目錄 `requirements.txt`，主要包含 Streamlit、pandas、numpy、yfinance、requests、PyTorch、scikit-learn、Plotly 與 Gemini SDK。建議使用 Python 3.8+ 虛擬環境。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Windows 也可使用 `start_simulation_system.bat`，但仍應確認批次檔的目前工作目錄與 Python 解譯器設定。

## API 設定

Gemini 服務會依序尋找：

1. `GEMINI_API_KEY`
2. `GOOGLE_GEMINI_API_KEY`
3. Streamlit secrets 的 `GEMINI_API_KEY`

不要把金鑰寫入 Python、JSON、CSV 或提交到版本控制。沒有金鑰時 `LLMService` 會回傳可讀的停用訊息，其他技術分析流程仍可能運作。

## 測試

目前可見的測試/診斷入口包括根目錄的 `test_etf_display.py`，以及 `scripts/` 中的快取、Parquet 與效能工具。修改後至少應執行：

```powershell
python -m compileall app.py ai_advisor.py core services jobs pages_ui
python test_etf_display.py
```

若測試需要網路或外部 API，應明確區分資料連線失敗與程式邏輯失敗。

## 修改檢查表

- 是否改到了正確層級，而不是在 UI 複製 service 邏輯？
- 是否維持 `AIAdvisor`、`PaperAccount` 與 trader 的既有公開方法格式？
- 是否處理空 DataFrame、缺少指標、網路 timeout 與 API key 缺失？
- 是否避免把真實交易語意誤寫成模擬交易已成交？
- 是否同步檢查相對路徑、帳戶檔、模型檔與快取格式？
- 是否新增或更新對應的測試？

## 文件維護

當新增核心模組、改變資料流、改變持久化格式或移除公開方法時，請同步更新 `docs/README.md` 的索引與本目錄相關文件。文件應描述目前實作，不要只保留歷史版本的產品宣稱。
