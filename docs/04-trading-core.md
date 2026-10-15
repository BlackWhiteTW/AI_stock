# 交易核心

## `core/trader.py`：PaperAccount

`PaperAccount` 是所有模擬交易器共用的帳戶實作，不連接券商。

### 買入

`buy(symbol, price, qty)` 會計算台股手續費 `max(20, cost * 0.1425%)`，檢查餘額，更新持倉平均成本，寫入 BUY history 並保存 JSON。

### 賣出

`sell(symbol, price, qty)` 會檢查持倉，扣除手續費 `0.1425%` 與證交稅 `0.3%`，更新餘額及持倉，寫入 SELL history 並保存 JSON。

### 股息

`check_dividends()` 每日最多執行一次，透過 Yahoo Finance 查詢近一個月股息並依台股/美股規則估算實領金額。

### 檔案格式

預設檔案：`data/account_data.json`。AI 交易器會使用自己的帳戶檔，以免污染使用者主帳戶。

## `core/auto_trader.py`：V1

V1 以單一標的回測與交易為主，包含：

- ATR 動態停損/停利。
- 固定百分比停損、停利與移動停損備援。
- 分批建倉與單一持股比例上限。
- `PerformanceTracker` 交易績效記錄。

`run_simulation()` 會從足夠計算 MA60 的位置開始，使用當日收盤分析、隔日開盤成交，避免把未來價格提前帶入決策。

## `core/auto_trader_v2.py`：V2

V2 是多標的、多訊號版本，主要能力：

- 技術分析 70% + 市場情緒 30% 的信號融合。
- 依分數、信心度、波動度與持倉數量動態計算倉位。
- 最多同時持有 5 檔，並使用相關性門檻避免集中風險。
- 偵測 `BULL`、`BEAR`、`VOLATILE`、`NORMAL` 市場狀態。
- 使用 ATR 停損停利、移動停損與最大回撤風控。

V2 預設帳戶檔是 `data/ai_account.json`；但 `app.py` 建立 V2 實例時指定 `data/ai_account_v2.json`，修改時請保留此差異。

## 測試與修改規則

交易規則改動應至少測試：餘額不足、持倉不足、手續費/稅費、加碼平均成本、全部賣出、回測隔日成交與資料不足。不要用真實帳戶 JSON 做測試資料。
