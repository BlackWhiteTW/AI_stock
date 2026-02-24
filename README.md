# AI 智慧台股投資模擬系統 (AI Stock Investment Simulation System)

這是一個基於 Python 開發的綜合性台股分析與自動化交易模擬平台。系統整合了機器學習預測、技術指標分析、市場情緒監控以及財務報表數據，旨在提供投資者科學化的決策支持。

## 🌟 核心功能

-   **AI 預測引擎**：利用深度學習與機器學習模型，針對台股進行每日走勢預測。
-   **自動化交易模擬**：內建 `AutoTrader V2`，模擬真實市場環境下的自動跟單與策略執行。
-   **多維度數據整合**：
    -   **技術面**：整合常用技術指標（KDJ, RSI, MACD 等）。
    -   **基本面**：串接 MOPS 與 FinMind API 獲取財務報表與除權息資訊。
    -   **消息面**：新聞爬蟲與 AI 情緒分析（News Analyzer）。
-   **視覺化儀表板**：基於 Streamlit 開發的多功能 UI，包含資產管理、策略回測與 AI 建議。
-   **自動化排程**：每日自動更新市場數據、重新訓練模型並產出預測報告。

## 📁 專案結構導覽

-   `app.py`: 系統主程式入口（Streamlit UI）。
-   `core/`: 交易核心邏輯，包含模擬交易員與自動交易策略。
-   `services/`: 後端服務層，負責資料爬取（Yahoo Finance, FinMind）、技術分析與 AI 預測模型。
-   `pages_ui/`: UI 介面模組化程式碼。
-   `jobs/`: 排程任務，如每日數據更新與模型訓練。
-   `scripts/`: 系統維護、模型優化與資料遷移工具。
-   `data/`: 存放歷史數據、模型檔案、交易日誌與設定檔。

## 🛠️ 安裝與啟動

### 1. 環境需求
-   Python 3.8+
-   建議使用虛擬環境 (Virtual Environment)

### 2. 安裝依賴
```bash
pip install -r requirements.txt
```

### 3. 啟動系統
在 Windows 環境下，您可以直接執行批次檔：
```bash
start_simulation_system.bat
```
或者手動啟動 Streamlit：
```bash
streamlit run app.py
```

## 📊 模型訓練
若要手動更新 AI 模型，可使用以下腳本：
- `python scripts/train_base_model.py`: 訓練基礎預測模型。
- `python scripts/advanced_trainer.py`: 執行進階模型優化。

## 📝 免責聲明
本系統僅供學術研究與投資模擬使用，所提供的 AI 預測結果不代表任何投資建議。投資人應獨立思考並自行承擔市場風險。
