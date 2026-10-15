# AI 與策略

## `ai_advisor.py`：外觀層

`AIAdvisor` 是 UI、jobs 與交易器使用的統一入口。它本身主要負責組合與代理：

- `AIService`：模型、預測、每日預測/驗證。
- `StockStrategy`：技術指標策略、股票分析、參數最佳化。
- `NewsAnalyzer`：新聞情緒。
- `tech_analysis`：技術指標計算。

新增 AI 功能時，優先放在對應 service，再在 `AIAdvisor` 暴露穩定的代理方法。

## `services/tech_analysis.py`

`calculate_technical_indicators()` 會在行情 DataFrame 上加入：

- 趨勢與動能：`RSI`, `MACD`, `MA20`, `MA_Long`, `BIAS_20`
- KDJ/成交量：`K`, `D`, `OBV`
- 波動：`ATR`, `Volatility`, `ATR_Ratio`
- Alpha：`Price_Velocity`, `Price_Acceleration`, `Relative_Strength`, `Dist_MA20`
- 可選財務特徵：`fin_*`

預設參數在 `DEFAULT_PARAMS`，包含 RSI 門檻、短均線與長均線。函式也會使用指標快取，避免相同資料重算。

## `services/stock_strategy.py`

`StockStrategy.analyze_stock()` 的決策順序大致為：

1. 檢查資料量與流動性。
2. 判斷均線趨勢。
3. 評估 RSI、MACD、成交量與位階。
4. 若有 `ai_pred_price`，加入預測偏差分數。
5. 若啟用新聞，加入新聞情緒分數。
6. 依總分產生買進、賣出、觀望或資料不足等 action。

`optimize_strategy_basic()` 以數組參數執行簡單回測，最佳參數只存在目前實例的 `learned_params`，不會自動寫檔。

## `services/ai_service.py`

負責：

- 掃描 `data/saved_models/` 載入模型 metadata。
- 以 CPU/GPU 自動選擇 PyTorch device。
- 維護最多 15 個 loaded model 的記憶體快取。
- 執行模型訓練、預測、每日預測與收盤驗證。
- 將訓練記錄寫入 `data/training_records/`，每日預測寫入 `data/daily_predictions.json`。

## 版本一致性提醒

目前 UI 文案宣稱 LSTM 已移除，但 `AIService` 仍保留 `LSTMModel`、`HybridTransLSTM`、`train_deep_learning_model()` 與 `predict_next_price()`。因此：

- 不要直接刪除模型相容程式，除非同時確認舊模型檔與 jobs 不再使用。
- 修改預測功能前，檢查 `AIAdvisor.predict_next_price()`、每日預測與 UI 分析頁的實際呼叫。
- 文件描述以呼叫路徑和檔案現況為準，UI 公告不等於整個 service 已移除。
