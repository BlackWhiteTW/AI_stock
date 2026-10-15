# 系統總覽

## 目的

本專案以 Python + Streamlit 建立台股分析與模擬交易平台，整合歷史行情、技術指標、財務資料、新聞情緒、AI 服務與虛擬帳戶。

## 分層

```text
app.py
  -> pages_ui/                 Streamlit 畫面與互動
  -> ai_advisor.py             AI 顧問外觀層
       -> services/ai_service.py
       -> services/stock_strategy.py
       -> services/news_analyzer.py
  -> core/                     虛擬帳戶與自動交易
  -> jobs/                     早盤/收盤與批次任務
  -> utils.py / services/      資料取得、分析、快取
  -> data/                     本地資料、模型、交易帳戶與報告
```

## 主要資料流

1. `utils.get_stock_data()` 透過 `MarketDataManager` 取得行情，必要時從 Yahoo Finance 補抓並寫入本地快取。
2. `AIAdvisor.analyze_stock()` 先取得預測價格，再交給 `StockStrategy.analyze_stock()` 將技術指標、流動性、新聞與預測整合成 `action / score / reasons`。
3. `AutoTrader` 或 `AutoTraderV2` 以分析結果決定買賣數量，透過 `PaperAccount` 模擬成交並保存交易紀錄。
4. Streamlit UI 從 `app.py` 組合側邊欄與分頁；背景執行緒負責平日 08:30 與 14:30 任務。

## 關鍵契約

### 行情 DataFrame

分析流程通常需要 `Open`, `High`, `Low`, `Close`, `Volume` 欄位，並以日期索引排序。技術指標會額外加入 `RSI`, `MACD`, `MA20`, `MA_Long`, `ATR`, `K`, `D`, `OBV` 等欄位。

### 分析結果

`StockStrategy.analyze_stock()` 的主要回傳格式：

```python
{
    "action": str,
    "score": int | float,
    "reasons": list[str],
    "price": float,
}
```

### 帳戶資料

`PaperAccount` 使用 `balance`, `portfolio`, `history`, `last_dividend_check` 四個主要欄位。交易成功後會立即寫回 JSON。

## 修改時的判斷順序

先確認改動屬於資料取得、分析、交易執行、UI 或排程中的哪一層，再修改該層；避免在 UI 中重寫策略，或在策略中直接操作 Streamlit `session_state`。
