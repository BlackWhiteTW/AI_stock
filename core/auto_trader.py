import pandas as pd
import datetime
from .trader import PaperAccount
from ai_advisor import AIAdvisor
from services.performance_tracker import PerformanceTracker

class AutoTrader:
    def __init__(self, account_file='data/ai_account.json', initial_balance=1000000):
        # 這是專屬於 AI 的帳戶，與使用者主帳戶分開
        self.account = PaperAccount(data_file=account_file, initial_balance=initial_balance)
        self.advisor = AIAdvisor()
        
        # 🆕 績效追蹤系統
        self.performance_tracker = PerformanceTracker()
        
        # 🆕 v4.0 動態風控參數（基於 ATR）
        self.use_dynamic_stops = True        # 啟用動態停損
        self.atr_stop_multiplier = 2.5      # ATR 停損倍數
        self.atr_profit_multiplier = 4.0    # ATR 停利倍數
        
        # 固定停損停利（備用）
        self.stop_loss_pct = -0.08          # 停損: -8%
        self.take_profit_pct = 0.15         # 停利: +15%
        self.trailing_stop_pct = 0.05       # 移動停損: 從最高點回落 5%
        self.max_position_ratio = 0.30      # 單一持股最多 30% 資金
        
        # 記錄每個持股的最高價 (用於移動停損)
        self.position_peaks = {}  # {ticker: highest_price}
        
    def reset(self):
        self.account.reset_account()
        self.position_peaks = {}  # 重置最高價記錄
        
    def run_simulation(self, ticker, df, progress_callback=None, lookback_days=60):
        """
        執行歷史回測模擬 (模擬真實交易，包含手續費)
        :param progress_callback: func(current_idx, total_len, current_date, assets)
        :param lookback_days: 回測天數 (從幾天前開始)
        """
        self.reset()
        
        # 為了模擬真實情況，我們必須一天一天推進
        total_days = len(df)
        
        # 用戶指定：從 N 天前開始回測 (加快速度並聚焦近期表現)
        # 但需確保至少有足夠的 K 線來計算 MA60 (至少留 70 天給前面)
        lookback_limit = lookback_days
        start_idx = max(70, total_days - lookback_limit)
        
        if total_days < start_idx:
            return "數據不足", pd.DataFrame()

        # 🚀 效能優化：預先計算所有技術指標
        # 避免在迴圈中重複計算 (O(N) vs O(N^2))
        try:
            # 取得該股票設定參數
            params = self.advisor._get_params(ticker)
            # 計算指標並更新到 df
            df = self.advisor.calculate_technical_indicators(df.copy(), params)
        except Exception as e:
            print(f"預先計算指標失敗: {e}")

        logs = []
        
        # 每天收盤後進行分析與下單 (模擬次日開盤成交)
        for i in range(start_idx, total_days-1):
            # 取得截至當日(i)的歷史數據
            current_df = df.iloc[:i+1].copy()
            today_date = current_df.index[-1]
            
            # 今日數據
            today_close = current_df['Close'].iloc[-1]
            
            # 🆕 更新最高價記錄 (用於移動停損)
            if ticker in self.account.portfolio and self.account.portfolio[ticker]['qty'] > 0:
                if ticker not in self.position_peaks:
                    self.position_peaks[ticker] = today_close
                else:
                    self.position_peaks[ticker] = max(self.position_peaks[ticker], today_close)
            
            # 🆕 v4.0 動態風險管理：檢查停損停利
            portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
            if portfolio_qty > 0:
                avg_cost = self.account.portfolio[ticker]['avg_price']
                profit_loss_pct = (today_close - avg_cost) / avg_cost
                
                # 🆕 計算動態停損停利（基於 ATR）
                if self.use_dynamic_stops and 'ATR' in current_df.columns:
                    current_atr = current_df['ATR'].iloc[-1]
                    
                    # 動態停損 = 當前價 - (2.5 × ATR)
                    dynamic_stop_price = today_close - (self.atr_stop_multiplier * current_atr)
                    dynamic_stop_pct = (dynamic_stop_price - avg_cost) / avg_cost
                    
                    # 動態停利 = 當前價 + (4.0 × ATR)
                    dynamic_profit_price = avg_cost + (self.atr_profit_multiplier * current_atr)
                    dynamic_profit_pct = (dynamic_profit_price - avg_cost) / avg_cost
                    
                    stop_loss_threshold = dynamic_stop_pct
                    take_profit_threshold = dynamic_profit_pct
                else:
                    # 使用固定百分比
                    stop_loss_threshold = self.stop_loss_pct
                    take_profit_threshold = self.take_profit_pct
                
                # 檢查停損
                if profit_loss_pct <= stop_loss_threshold:
                    # 強制停損
                    next_day_row = df.iloc[i+1]
                    next_day_open = next_day_row['Open']
                    success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                    if success:
                        log_date = next_day_row.name
                        log_close = next_day_row['Close']
                        total_assets = self.account.balance
                        
                        # 🆕 記錄到績效追蹤
                        self.performance_tracker.log_trade(
                            ticker, 'SELL', next_day_open, portfolio_qty, 
                            log_date, f'停損 ({profit_loss_pct:.1%})'
                        )
                        
                        logs.append({
                            "Date": log_date,
                            "Price": log_close,
                            "Action": "STOP_LOSS",
                            "DealPrice": next_day_open,
                            "Assets": total_assets,
                            "Signal": f"🛑 {'動態' if self.use_dynamic_stops else '固定'}停損 ({profit_loss_pct:.1%})",
                            "Score": -10
                        })
                        
                        # 清除最高價記錄
                        if ticker in self.position_peaks:
                            del self.position_peaks[ticker]
                        
                        if progress_callback:
                            date_str = log_date.strftime("%Y-%m-%d")
                            progress_callback(i - start_idx + 1, total_days - 1 - start_idx, date_str, total_assets)
                        continue  # 跳過本次一般分析
                
                # 檢查停利
                if profit_loss_pct >= take_profit_threshold:
                    # 獲利了結
                    next_day_row = df.iloc[i+1]
                    next_day_open = next_day_row['Open']
                    success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                    if success:
                        log_date = next_day_row.name
                        log_close = next_day_row['Close']
                        total_assets = self.account.balance
                        
                        # 🆕 記錄到績效追蹤
                        self.performance_tracker.log_trade(
                            ticker, 'SELL', next_day_open, portfolio_qty,
                            log_date, f'停利 ({profit_loss_pct:.1%})'
                        )
                        
                        logs.append({
                            "Date": log_date,
                            "Price": log_close,
                            "Action": "TAKE_PROFIT",
                            "DealPrice": next_day_open,
                            "Assets": total_assets,
                            "Signal": f"✅ {'動態' if self.use_dynamic_stops else '固定'}停利 ({profit_loss_pct:.1%})",
                            "Score": 10
                        })
                        
                        if ticker in self.position_peaks:
                            del self.position_peaks[ticker]
                        
                        if progress_callback:
                            date_str = log_date.strftime("%Y-%m-%d")
                            progress_callback(i - start_idx + 1, total_days - 1 - start_idx, date_str, total_assets)
                        continue
                
                # 檢查移動停損 (已獲利 > 5% 時啟動)
                if profit_loss_pct > 0.05 and ticker in self.position_peaks:
                    peak_price = self.position_peaks[ticker]
                    drawdown_from_peak = (today_close - peak_price) / peak_price
                    
                    if drawdown_from_peak <= -self.trailing_stop_pct:
                        # 觸發移動停損
                        next_day_row = df.iloc[i+1]
                        next_day_open = next_day_row['Open']
                        success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                        if success:
                            log_date = next_day_row.name
                            log_close = next_day_row['Close']
                            total_assets = self.account.balance
                            
                            logs.append({
                                "Date": log_date,
                                "Price": log_close,
                                "Action": "TRAILING_STOP",
                                "DealPrice": next_day_open,
                                "Assets": total_assets,
                                "Signal": f"📉 移動停損 (高點回落 {abs(drawdown_from_peak):.1%})",
                                "Score": 5
                            })
                            
                            del self.position_peaks[ticker]
                            
                            if progress_callback:
                                date_str = log_date.strftime("%Y-%m-%d")
                                progress_callback(i - start_idx + 1, total_days - 1 - start_idx, date_str, total_assets)
                            continue
            
            # 取得「明日」數據 (真實交易發生的時間點)
            # 因為我們是在今日收盤後決定策略，並掛單在明日開盤成交
            next_day_row = df.iloc[i+1]
            next_day_open = next_day_row['Open']
            next_day_change = (next_day_row['Close'] - next_day_row['Open']) # 用於紀錄當日波動
            
            # 1. 呼叫 AI 投顧進行分析 (基於今日收盤資訊)
            # 注意：這裡會觸發 AI 運算，包含技術指標計算
            # 歷史回測時不抓取網路新聞 (include_news=False)
            analysis = self.advisor.analyze_stock(ticker, current_df, include_news=False)
            
            action_signal = analysis['action']
            ai_score = analysis['score']
            
            # 2. 執行交易策略 (以明日開盤價成交)
            # 模擬真實情境：今日晚上掛單，明日開盤撮合
            execution_price = next_day_open
            
            cash = self.account.balance
            portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
            
            trade_action = None
            trade_qty = 0
            
            # 買入邏輯
            if "買進" in action_signal and cash > execution_price * 1000:
                # 🆕 改進：分批建倉策略
                # 計算目標倉位比例
                if "強力" in action_signal or "強烈" in action_signal:
                    target_position_ratio = 0.60  # 強力買進: 目標 60% 資金
                else:
                    target_position_ratio = 0.30  # 一般買進: 目標 30% 資金
                
                # 計算總資產
                total_assets = cash + sum(
                    self.account.portfolio[s]['qty'] * execution_price 
                    for s in self.account.portfolio
                )
                
                # 目標倉位金額
                target_position_value = total_assets * target_position_ratio
                
                # 目前該股票倉位金額
                current_position_value = portfolio_qty * execution_price
                
                # 還需要買入的金額
                needed_amount = target_position_value - current_position_value
                
                if needed_amount > 0:
                    # 🆕 分批進場: 每次買入剩餘目標的 40%
                    # 這樣可以分 3 批: 40% -> 24% -> 14% ... 逐步建倉
                    batch_ratio = 0.4
                    buy_amount = needed_amount * batch_ratio
                    
                    # 確保至少有 1000 元以上才買 (避免太小的交易)
                    if buy_amount >= 1000:
                        qty_to_buy = int(buy_amount // execution_price)
                        
                        if qty_to_buy > 0:
                            success, msg = self.account.buy(ticker, execution_price, qty_to_buy)
                            if success:
                                trade_action = "BUY"
                                trade_qty = qty_to_buy
            
            # 賣出邏輯
            elif "賣出" in action_signal and portfolio_qty > 0:
                # 若強烈賣出則清倉，否則減碼一半
                ratio = 1.0 if "強烈" in action_signal else 0.5
                qty_to_sell = int(portfolio_qty * ratio)
                
                if qty_to_sell > 0:
                    # 注意：這裡使用的是 execution_price (明日開盤價)
                    success, msg = self.account.sell(ticker, execution_price, qty_to_sell)
                    if success:
                        trade_action = "SELL"
                        trade_qty = qty_to_sell
            
            # 3. 紀錄資產狀況 (使用當日收盤價結算市值，但交易發生在開盤)
            # 這裡的 "Price" 紀錄的是當天(i)的收盤，還是次日(i+1)的？
            # 為了對應 date，我們應該紀錄的是 "執行當日(i+1)" 的狀況
            # 因為交易是在 i+1 早上發生的
            
            # 更新邏輯：為了讓圖表對齊，我們將這筆 Log 視為 i+1 的紀錄
            log_date = next_day_row.name # i+1 的日期
            log_close = next_day_row['Close'] # i+1 的收盤價 (用於計算當日結算市值)
            
            # 交易後的庫存 (account 已經更新了)
            current_portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
            market_value = current_portfolio_qty * log_close
            total_assets = self.account.balance + market_value
            
            logs.append({
                "Date": log_date,
                "Price": log_close, # 紀錄當日收盤價
                "Action": trade_action if trade_action else "",
                "DealPrice": execution_price if trade_action else None, # 紀錄成交價(開盤價)
                "Assets": total_assets,
                "Signal": action_signal,
                "Score": ai_score
            })
            
            # 回調進度
            if progress_callback:
                # 將 date 轉字串
                date_str = log_date.strftime("%Y-%m-%d")
                progress_callback(i - start_idx + 1, total_days - 1 - start_idx, date_str, total_assets)
                
        return "模擬完成", pd.DataFrame(logs).set_index("Date")

    def execute_daily_trade(self, ticker, df):
        """
        執行單日真實模擬交易 (Paper Trading)
        適合每天收盤後執行一次，根據當日收盤價決定是否下單
        """
        if df.empty: return "無數據", None
        
        # 使用最新的數據進行分析
        today_date = df.index[-1] # Timestamp
        today_close = df['Close'].iloc[-1]
        
        # 🆕 更新最高價記錄
        if ticker in self.account.portfolio and self.account.portfolio[ticker]['qty'] > 0:
            if ticker not in self.position_peaks:
                self.position_peaks[ticker] = today_close
            else:
                self.position_peaks[ticker] = max(self.position_peaks[ticker], today_close)
        
        # 🆕 風險管理：檢查停損停利
        portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
        if portfolio_qty > 0:
            avg_cost = self.account.portfolio[ticker]['avg_price']
            profit_loss_pct = (today_close - avg_cost) / avg_cost
            
            # 檢查固定停損
            if profit_loss_pct <= self.stop_loss_pct:
                success, msg = self.account.sell(ticker, today_close, portfolio_qty)
                if success:
                    if ticker in self.position_peaks:
                        del self.position_peaks[ticker]
                    return f"🛑 {ticker} 觸發停損 ({profit_loss_pct:.1%})，已全部賣出", {"action": "STOP_LOSS", "score": -10}
            
            # 檢查固定停利
            if profit_loss_pct >= self.take_profit_pct:
                success, msg = self.account.sell(ticker, today_close, portfolio_qty)
                if success:
                    if ticker in self.position_peaks:
                        del self.position_peaks[ticker]
                    return f"✅ {ticker} 達成獲利目標 ({profit_loss_pct:.1%})，已全部賣出", {"action": "TAKE_PROFIT", "score": 10}
            
            # 檢查移動停損
            if profit_loss_pct > 0.05 and ticker in self.position_peaks:
                peak_price = self.position_peaks[ticker]
                drawdown_from_peak = (today_close - peak_price) / peak_price
                
                if drawdown_from_peak <= -self.trailing_stop_pct:
                    success, msg = self.account.sell(ticker, today_close, portfolio_qty)
                    if success:
                        del self.position_peaks[ticker]
                        return f"📉 {ticker} 觸發移動停損 (從高點回落 {abs(drawdown_from_peak):.1%})，已全部賣出", {"action": "TRAILING_STOP", "score": 5}
        
        # 檢查是否今日已經交易過 (避免重複下單)
        # 這裡簡單檢查最後一筆交易紀錄的日期
        last_trade_date = None
        if self.account.history:
            last_trade_time_str = self.account.history[0]['time'] # "2024-05-20 13:00:00"
            last_trade_date = last_trade_time_str.split(" ")[0]
        
        # 將 Timestamp 轉字串比較
        today_str = today_date.strftime("%Y-%m-%d")
        
        if last_trade_date == today_str:
            return f"⚠️ 今日 ({today_str}) 已經執行過交易，跳過。", None

        # 呼叫 AI 分析
        analysis = self.advisor.analyze_stock(ticker, df)
        action_signal = analysis['action']
        ai_score = analysis['score']
        
        cash = self.account.balance
        portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
        
        trade_action = "HOLD"
        trade_details = ""
        
        # 買入邏輯
        if "買進" in action_signal and cash > today_close * 1000:
            ratio = 0.8 if "強烈" in action_signal else 0.4
            budget = cash * ratio
            qty_to_buy = int(budget // today_close)
            
            if qty_to_buy > 0:
                success, msg = self.account.buy(ticker, today_close, qty_to_buy)
                if success:
                    trade_action = "BUY"
                    trade_details = msg
        
        # 賣出邏輯
        elif "賣出" in action_signal and portfolio_qty > 0:
            ratio = 1.0 if "強烈" in action_signal else 0.5
            qty_to_sell = int(portfolio_qty * ratio)
            
            if qty_to_sell > 0:
                success, msg = self.account.sell(ticker, today_close, qty_to_sell)
                if success:
                    trade_action = "SELL"
                    trade_details = msg
        
        # 強制存檔 (雖然 buy/sell 會存，但如果沒交易也要確保狀態)
        self.account.save_data()
        
        return f"📅 {today_str} AI 執行結果: {action_signal} (Score: {ai_score}) -> {trade_action} {trade_details}", analysis


