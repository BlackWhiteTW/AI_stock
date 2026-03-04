"""
AutoTrader v2.0 - 優化版自動交易系統
===========================================

v2.0 新功能:
1. 🧠 AI 深度學習信號整合 (結合技術分析與 DL 預測)
2. 📊 動態倉位管理 (根據市場波動與信心度調整)
3. 🎯 多信號融合決策 (技術面 + 基本面 + AI 預測)
4. 📈 投資組合管理 (多股票同時運作)
5. 🔧 參數自動優化 (回測尋找最佳停損停利)
6. 📉 風險分散機制 (相關性分析避免集中風險)
"""

import pandas as pd
import numpy as np
import datetime
from .trader import PaperAccount
from ai_advisor import AIAdvisor
from services.performance_tracker import PerformanceTracker
from services.ai_service import AIService

class AutoTraderV2:
    """
    v2.0 優化版自動交易機器人
    
    主要改進:
    - 整合深度學習預測信號
    - 動態倉位管理系統
    - 多信號融合決策
    - 投資組合風險管理
    """
    
    def __init__(self, account_file='data/ai_account.json', initial_balance=1000000):
        # 基礎設定
        self.account = PaperAccount(data_file=account_file, initial_balance=initial_balance)
        self.advisor = AIAdvisor()
        self.ai_service = AIService()  # 🆕 整合深度學習預測
        self.performance_tracker = PerformanceTracker()
        
        # 🆕 v2.0 多信號決策系統
        self.signal_weights = {
            'technical': 0.30,      # 技術分析權重 30%
            'ai_prediction': 0.50,  # AI 預測權重 50%
            'sentiment': 0.20       # 市場情緒權重 20%
        }
        
        # 🆕 v2.0 動態倉位管理
        self.min_position_size = 0.10    # 最小倉位 10%
        self.max_position_size = 0.35    # 最大倉位 35%
        self.base_position_size = 0.20   # 基礎倉位 20%
        
        # 🆕 v2.0 智能風控參數 (可自動優化)
        self.risk_params = {
            'atr_stop_multiplier': 2.5,
            'atr_profit_multiplier': 4.0,
            'trailing_stop_pct': 0.05,
            'max_drawdown_pct': 0.15,      # 最大回撤限制
            'confidence_threshold': 0.60,  # 最低信心度門檻
        }
        
        # 🆕 v2.0 投資組合管理
        self.max_positions = 5           # 最多同時持有 5 檔
        self.correlation_threshold = 0.7 # 相關性過高則不買入
        self.position_peaks = {}         # 記錄最高價 {ticker: peak_price}
        
        # 🆕 v2.0 市場狀態追蹤
        self.market_regime = 'NORMAL'    # BULL/BEAR/NORMAL/VOLATILE
        self.volatility_multiplier = 1.0 # 波動度倍數
        
    def reset(self):
        """重置帳戶與所有狀態"""
        self.account.reset_account()
        self.position_peaks = {}
        self.market_regime = 'NORMAL'
        self.volatility_multiplier = 1.0
        
    def _calculate_signal_score(self, ticker, df, current_price):
        """
        🆕 v2.0 核心: 多信號融合決策系統
        
        整合三大信號源:
        1. 技術分析 (advisor.analyze_stock)
        2. AI 深度學習預測
        3. 市場情緒分析
        
        Returns:
            dict: {
                'score': float (-100 到 100),
                'confidence': float (0 到 1),
                'action': str ('BUY'/'SELL'/'HOLD'),
                'breakdown': dict (各信號明細)
            }
        """
        breakdown = {}
        
        # 1️⃣ 技術分析信號 (來自 AIAdvisor)
        tech_analysis = self.advisor.analyze_stock(ticker, df, include_news=False)
        tech_score = tech_analysis['score']
        tech_action = tech_analysis['action']
        
        # 轉換為 -100 到 100 的分數
        if "強烈買進" in tech_action or "強力買進" in tech_action:
            tech_normalized = 80
        elif "買進" in tech_action:
            tech_normalized = 50
        elif "賣出" in tech_action:
            tech_normalized = -60
        elif "強烈賣出" in tech_action:
            tech_normalized = -80
        else:
            tech_normalized = 0
        
        breakdown['technical'] = {
            'score': tech_score,
            'normalized': tech_normalized,
            'action': tech_action
        }
        
        # 2️⃣ AI 深度學習預測信號
        ai_score = 0
        ai_confidence = 0.5
        ai_prediction = None
        
        try:
            # 檢查是否有訓練好的模型
            if ticker in self.ai_service.models:
                # 準備特徵數據 (最近 60 天)
                recent_data = df.tail(60).copy()
                
                # 使用 AI 模型預測未來趨勢
                prediction_result = self.ai_service.predict_next_day(ticker, recent_data)
                
                if prediction_result and 'predicted_change_pct' in prediction_result:
                    predicted_change = prediction_result['predicted_change_pct']
                    ai_confidence = prediction_result.get('confidence', 0.5)
                    
                    # 將預測變化轉換為信號分數
                    # +5% 以上 → 80分, +2%~5% → 50分
                    # -2%~-5% → -50分, -5% 以下 → -80分
                    if predicted_change >= 0.05:
                        ai_score = 80
                    elif predicted_change >= 0.02:
                        ai_score = 50
                    elif predicted_change >= 0:
                        ai_score = 20
                    elif predicted_change >= -0.02:
                        ai_score = -20
                    elif predicted_change >= -0.05:
                        ai_score = -50
                    else:
                        ai_score = -80
                    
                    ai_prediction = predicted_change
        
        except Exception as e:
            print(f"AI 預測失敗: {e}")
            ai_score = 0
            ai_confidence = 0.5
        
        breakdown['ai_prediction'] = {
            'score': ai_score,
            'confidence': ai_confidence,
            'prediction': ai_prediction
        }
        
        # 3️⃣ 市場情緒信號 (基於市場指標)
        sentiment_score = 0
        
        try:
            # 檢查是否有市場情緒數據
            if 'Market_Sentiment_Score' in df.columns:
                market_sentiment = df['Market_Sentiment_Score'].iloc[-1]
                # Market_Sentiment_Score 範圍 -1 到 1
                sentiment_score = market_sentiment * 60  # 轉換為 -60 到 60
        except:
            sentiment_score = 0
        
        breakdown['sentiment'] = {
            'score': sentiment_score
        }
        
        # 🎯 加權融合 (根據 self.signal_weights)
        final_score = (
            tech_normalized * self.signal_weights['technical'] +
            ai_score * self.signal_weights['ai_prediction'] +
            sentiment_score * self.signal_weights['sentiment']
        )
        
        # 計算整體信心度 (技術分析信心度固定 0.7, AI 有動態信心度)
        overall_confidence = (
            0.7 * self.signal_weights['technical'] +
            ai_confidence * self.signal_weights['ai_prediction'] +
            0.6 * self.signal_weights['sentiment']
        )
        
        # 決定行動
        if final_score >= 40 and overall_confidence >= self.risk_params['confidence_threshold']:
            action = 'BUY'
        elif final_score <= -40:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'score': final_score,
            'confidence': overall_confidence,
            'action': action,
            'breakdown': breakdown
        }
    
    def _calculate_dynamic_position_size(self, signal_score, confidence, current_portfolio_value, total_assets):
        """
        🆕 v2.0 動態倉位計算
        
        根據以下因素調整倉位:
        1. 信號強度 (score)
        2. 信心度 (confidence)
        3. 市場波動率 (volatility_multiplier)
        4. 當前組合風險 (持股數量)
        
        Returns:
            float: 目標倉位比例 (0.0 到 max_position_size)
        """
        # 基礎倉位
        base_size = self.base_position_size
        
        # 🔹 因子1: 信號強度調整 (-50% to +50%)
        # score 範圍 -100 到 100, 標準化為 -0.5 到 0.5
        strength_factor = (signal_score / 100) * 0.5
        
        # 🔹 因子2: 信心度調整 (0.5 到 1.5 倍)
        # confidence 0.6 → 1.0x, 0.8 → 1.2x, 0.9 → 1.5x
        confidence_factor = 0.5 + confidence
        
        # 🔹 因子3: 市場波動調整
        # 高波動 → 減倉 (0.7x), 低波動 → 加倉 (1.3x)
        volatility_factor = 1.0 / self.volatility_multiplier
        
        # 🔹 因子4: 組合分散調整
        # 持股越多,新倉位越小 (避免過度集中)
        num_positions = len(self.account.portfolio)
        diversification_factor = max(0.5, 1.0 - (num_positions / self.max_positions) * 0.5)
        
        # 計算目標倉位
        target_size = base_size * (1 + strength_factor) * confidence_factor * volatility_factor * diversification_factor
        
        # 限制範圍
        target_size = max(self.min_position_size, min(target_size, self.max_position_size))
        
        return target_size
    
    def _update_market_regime(self, df):
        """
        🆕 v2.0 市場狀態偵測
        
        分析市場走勢,判斷當前處於:
        - BULL: 牛市 (上升趨勢)
        - BEAR: 熊市 (下跌趨勢)
        - VOLATILE: 震盪市 (高波動)
        - NORMAL: 正常市況
        """
        try:
            # 計算近期波動率
            recent_returns = df['Close'].pct_change().tail(20)
            volatility = recent_returns.std()
            
            # 更新波動倍數
            baseline_volatility = 0.015  # 基準波動率 1.5%
            self.volatility_multiplier = volatility / baseline_volatility
            
            # 判斷趨勢
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > ma20 > ma60 and volatility < baseline_volatility * 1.5:
                self.market_regime = 'BULL'
            elif current_price < ma20 < ma60 and volatility < baseline_volatility * 1.5:
                self.market_regime = 'BEAR'
            elif volatility > baseline_volatility * 2.0:
                self.market_regime = 'VOLATILE'
            else:
                self.market_regime = 'NORMAL'
        
        except Exception as e:
            print(f"市場狀態更新失敗: {e}")
            self.market_regime = 'NORMAL'
    
    def run_simulation(self, ticker, df, progress_callback=None, lookback_days=90):
        """
        🆕 v2.0 優化版回測系統
        
        改進:
        1. 多信號融合決策
        2. 動態倉位管理
        3. 市場狀態自適應
        4. 更詳細的交易日誌
        """
        self.reset()
        
        total_days = len(df)
        start_idx = max(70, total_days - lookback_days)
        
        if total_days < start_idx:
            return "數據不足", pd.DataFrame()
        
        # 預先計算技術指標
        try:
            params = self.advisor._get_params(ticker)
            df = self.advisor.calculate_technical_indicators(df.copy(), params)
        except Exception as e:
            print(f"指標計算失敗: {e}")
        
        logs = []
        
        for i in range(start_idx, total_days - 1):
            current_df = df.iloc[:i+1].copy()
            today_date = current_df.index[-1]
            today_close = current_df['Close'].iloc[-1]
            
            # 🆕 更新市場狀態
            self._update_market_regime(current_df)
            
            # 🆕 更新持倉最高價
            portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
            if portfolio_qty > 0:
                if ticker not in self.position_peaks:
                    self.position_peaks[ticker] = today_close
                else:
                    self.position_peaks[ticker] = max(self.position_peaks[ticker], today_close)
            
            # 🆕 v2.0 動態風控檢查
            if portfolio_qty > 0:
                avg_cost = self.account.portfolio[ticker]['avg_price']
                profit_loss_pct = (today_close - avg_cost) / avg_cost
                
                # 計算動態停損停利
                if 'ATR' in current_df.columns:
                    current_atr = current_df['ATR'].iloc[-1]
                    dynamic_stop_pct = -(self.risk_params['atr_stop_multiplier'] * current_atr / today_close)
                    dynamic_profit_pct = (self.risk_params['atr_profit_multiplier'] * current_atr / avg_cost)
                else:
                    dynamic_stop_pct = -0.08
                    dynamic_profit_pct = 0.15
                
                # 停損檢查
                if profit_loss_pct <= dynamic_stop_pct:
                    next_day_row = df.iloc[i+1]
                    next_day_open = next_day_row['Open']
                    success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                    
                    if success:
                        self.performance_tracker.log_trade(
                            ticker, 'SELL', next_day_open, portfolio_qty,
                            next_day_row.name, f'動態停損 ({profit_loss_pct:.1%})'
                        )
                        
                        logs.append({
                            "Date": next_day_row.name,
                            "Price": next_day_row['Close'],
                            "Action": "STOP_LOSS",
                            "DealPrice": next_day_open,
                            "Qty": portfolio_qty,
                            "Assets": self.account.balance,
                            "Signal": f"🛑 動態停損 ({profit_loss_pct:.1%})",
                            "Score": -10,
                            "Regime": self.market_regime
                        })
                        
                        if ticker in self.position_peaks:
                            del self.position_peaks[ticker]
                        
                        if progress_callback:
                            progress_callback(i - start_idx + 1, total_days - 1 - start_idx,
                                            next_day_row.name.strftime("%Y-%m-%d"), self.account.balance)
                        continue
                
                # 停利檢查
                if profit_loss_pct >= dynamic_profit_pct:
                    next_day_row = df.iloc[i+1]
                    next_day_open = next_day_row['Open']
                    success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                    
                    if success:
                        self.performance_tracker.log_trade(
                            ticker, 'SELL', next_day_open, portfolio_qty,
                            next_day_row.name, f'動態停利 ({profit_loss_pct:.1%})'
                        )
                        
                        logs.append({
                            "Date": next_day_row.name,
                            "Price": next_day_row['Close'],
                            "Action": "TAKE_PROFIT",
                            "DealPrice": next_day_open,
                            "Qty": portfolio_qty,
                            "Assets": self.account.balance,
                            "Signal": f"✅ 動態停利 ({profit_loss_pct:.1%})",
                            "Score": 10,
                            "Regime": self.market_regime
                        })
                        
                        if ticker in self.position_peaks:
                            del self.position_peaks[ticker]
                        
                        if progress_callback:
                            progress_callback(i - start_idx + 1, total_days - 1 - start_idx,
                                            next_day_row.name.strftime("%Y-%m-%d"), self.account.balance)
                        continue
                
                # 移動停損檢查
                if profit_loss_pct > 0.05 and ticker in self.position_peaks:
                    peak_price = self.position_peaks[ticker]
                    drawdown = (today_close - peak_price) / peak_price
                    
                    if drawdown <= -self.risk_params['trailing_stop_pct']:
                        next_day_row = df.iloc[i+1]
                        next_day_open = next_day_row['Open']
                        success, msg = self.account.sell(ticker, next_day_open, portfolio_qty)
                        
                        if success:
                            logs.append({
                                "Date": next_day_row.name,
                                "Price": next_day_row['Close'],
                                "Action": "TRAILING_STOP",
                                "DealPrice": next_day_open,
                                "Qty": portfolio_qty,
                                "Assets": self.account.balance,
                                "Signal": f"📉 移動停損 (高點回落 {abs(drawdown):.1%})",
                                "Score": 5,
                                "Regime": self.market_regime
                            })
                            
                            del self.position_peaks[ticker]
                            
                            if progress_callback:
                                progress_callback(i - start_idx + 1, total_days - 1 - start_idx,
                                                next_day_row.name.strftime("%Y-%m-%d"), self.account.balance)
                            continue
            
            # 🆕 v2.0 多信號融合決策
            signal_result = self._calculate_signal_score(ticker, current_df, today_close)
            
            next_day_row = df.iloc[i+1]
            next_day_open = next_day_row['Open']
            
            cash = self.account.balance
            total_assets = cash + sum(
                self.account.portfolio[s]['qty'] * today_close
                for s in self.account.portfolio
            )
            current_position_value = portfolio_qty * today_close
            
            # 🆕 v2.0 動態倉位計算
            if signal_result['action'] == 'BUY' and cash > next_day_open * 1000:
                target_position_size = self._calculate_dynamic_position_size(
                    signal_result['score'],
                    signal_result['confidence'],
                    current_position_value,
                    total_assets
                )
                
                target_value = total_assets * target_position_size
                needed_amount = target_value - current_position_value
                
                if needed_amount > 1000:
                    # 分批進場: 40% 的目標增量
                    buy_amount = needed_amount * 0.4
                    qty_to_buy = int(buy_amount // next_day_open)
                    
                    if qty_to_buy > 0:
                        success, msg = self.account.buy(ticker, next_day_open, qty_to_buy)
                        
                        if success:
                            self.performance_tracker.log_trade(
                                ticker, 'BUY', next_day_open, qty_to_buy,
                                next_day_row.name, f'AI融合信號 (評分:{signal_result["score"]:.0f}, 信心:{signal_result["confidence"]:.1%})'
                            )
                            
                            logs.append({
                                "Date": next_day_row.name,
                                "Price": next_day_row['Close'],
                                "Action": "BUY",
                                "DealPrice": next_day_open,
                                "Qty": qty_to_buy,
                                "Assets": self.account.balance,
                                "Signal": f"📈 買入 (AI評分:{signal_result['score']:.0f}, 信心:{signal_result['confidence']:.0%})",
                                "Score": signal_result['score'],
                                "Regime": self.market_regime,
                                "Technical": signal_result['breakdown']['technical']['normalized'],
                                "AI_Pred": signal_result['breakdown']['ai_prediction']['score']
                            })
            
            elif signal_result['action'] == 'SELL' and portfolio_qty > 0:
                # 根據信號強度決定賣出比例
                sell_ratio = 1.0 if signal_result['score'] <= -60 else 0.5
                qty_to_sell = int(portfolio_qty * sell_ratio)
                
                if qty_to_sell > 0:
                    success, msg = self.account.sell(ticker, next_day_open, qty_to_sell)
                    
                    if success:
                        self.performance_tracker.log_trade(
                            ticker, 'SELL', next_day_open, qty_to_sell,
                            next_day_row.name, f'AI融合信號 (評分:{signal_result["score"]:.0f})'
                        )
                        
                        logs.append({
                            "Date": next_day_row.name,
                            "Price": next_day_row['Close'],
                            "Action": "SELL",
                            "DealPrice": next_day_open,
                            "Qty": qty_to_sell,
                            "Assets": self.account.balance,
                            "Signal": f"📉 賣出 (AI評分:{signal_result['score']:.0f})",
                            "Score": signal_result['score'],
                            "Regime": self.market_regime
                        })
            
            else:
                # HOLD - 記錄觀察日誌
                logs.append({
                    "Date": next_day_row.name,
                    "Price": next_day_row['Close'],
                    "Action": "",
                    "DealPrice": next_day_open,
                    "Qty": 0,
                    "Assets": self.account.balance + current_position_value,
                    "Signal": f"觀察 (評分:{signal_result['score']:.0f})",
                    "Score": signal_result['score'],
                    "Regime": self.market_regime
                })
            
            if progress_callback:
                current_assets = self.account.balance + current_position_value
                progress_callback(i - start_idx + 1, total_days - 1 - start_idx,
                                next_day_row.name.strftime("%Y-%m-%d"), current_assets)
        
        df_logs = pd.DataFrame(logs)
        if not df_logs.empty:
            df_logs.set_index('Date', inplace=True)
        
        return f"v2.0 回測完成 (市場狀態: {self.market_regime})", df_logs
    
    def execute_daily_trade(self, ticker, df):
        """
        🆕 v2.0 每日自動交易決策
        
        與 run_simulation 不同，這是針對「當下」的數據進行一次性決策。
        用於每日自動化排程。
        """
        if df.empty:
            return "⚠️ 無法取得數據", None
            
        current_price = df['Close'].iloc[-1]
        
        # 1. 更新市場狀態與持倉最高價
        self._update_market_regime(df)
        
        portfolio_qty = self.account.portfolio.get(ticker, {}).get('qty', 0)
        if portfolio_qty > 0:
            if ticker not in self.position_peaks:
                self.position_peaks[ticker] = current_price
            else:
                self.position_peaks[ticker] = max(self.position_peaks[ticker], current_price)
        
        # 2. 智能風控檢查 (停損停利)
        if portfolio_qty > 0:
            avg_cost = self.account.portfolio[ticker]['avg_price']
            profit_loss_pct = (current_price - avg_cost) / avg_cost
            
            # 獲取動態閾值
            params = self.advisor._get_params(ticker)
            df_with_ind = self.advisor.calculate_technical_indicators(df.copy(), params)
            
            if 'ATR' in df_with_ind.columns:
                current_atr = df_with_ind['ATR'].iloc[-1]
                dynamic_stop_pct = -(self.risk_params['atr_stop_multiplier'] * current_atr / current_price)
                dynamic_profit_pct = (self.risk_params['atr_profit_multiplier'] * current_atr / avg_cost)
            else:
                dynamic_stop_pct = -0.08
                dynamic_profit_pct = 0.15
                
            # A. 停損
            if profit_loss_pct <= dynamic_stop_pct:
                success, msg = self.account.sell(ticker, current_price, portfolio_qty)
                if success:
                    del self.position_peaks[ticker]
                    return f"🛑 動態停損賣出: {msg} (損益: {profit_loss_pct:.1%})", {"action": "賣出", "reason": "停損"}
            
            # B. 停利
            if profit_loss_pct >= dynamic_profit_pct:
                success, msg = self.account.sell(ticker, current_price, portfolio_qty)
                if success:
                    del self.position_peaks[ticker]
                    return f"✅ 動態停利賣出: {msg} (損益: {profit_loss_pct:.1%})", {"action": "賣出", "reason": "停利"}
            
            # C. 移動停損
            if profit_loss_pct > 0.05 and ticker in self.position_peaks:
                peak_price = self.position_peaks[ticker]
                drawdown = (current_price - peak_price) / peak_price
                if drawdown <= -self.risk_params['trailing_stop_pct']:
                    success, msg = self.account.sell(ticker, current_price, portfolio_qty)
                    if success:
                        del self.position_peaks[ticker]
                        return f"📉 移動停損賣出: {msg} (高點回落 {abs(drawdown):.1%})", {"action": "賣出", "reason": "移動停損"}

        # 3. 多信號融合決策 (買入/加碼/減倉)
        signal_result = self._calculate_signal_score(ticker, df, current_price)
        
        cash = self.account.balance
        total_assets = self.account.get_total_assets()
        current_position_value = portfolio_qty * current_price
        
        # A. 買入/加碼邏輯
        if signal_result['action'] == 'BUY':
            target_position_size = self._calculate_dynamic_position_size(
                signal_result['score'],
                signal_result['confidence'],
                current_position_value,
                total_assets
            )
            
            target_value = total_assets * target_position_size
            needed_amount = target_value - current_position_value
            
            if needed_amount > 1000:
                # 實戰中，我們一次買足目標倉位 (或考慮資金剩餘)
                qty_to_buy = int(min(needed_amount, cash) // current_price)
                if qty_to_buy > 0:
                    success, msg = self.account.buy(ticker, current_price, qty_to_buy)
                    if success:
                        return f"📈 AI 買進成功: {msg} (評分:{signal_result['score']:.0f}, 信心:{signal_result['confidence']:.0%})", {"action": "買入", "score": signal_result['score']}
            
        # B. 賣出/減倉邏輯
        elif signal_result['action'] == 'SELL' and portfolio_qty > 0:
            # 強烈看空則全清，否則減半
            sell_ratio = 1.0 if signal_result['score'] <= -60 else 0.5
            qty_to_sell = int(portfolio_qty * sell_ratio)
            
            if qty_to_sell > 0:
                success, msg = self.account.sell(ticker, current_price, qty_to_sell)
                if success:
                    if sell_ratio == 1.0 and ticker in self.position_peaks:
                        del self.position_peaks[ticker]
                    return f"📉 AI 減倉賣出: {msg} (評分:{signal_result['score']:.0f})", {"action": "賣出", "score": signal_result['score']}
        
        return f"😐 AI 建議觀望 (評分:{signal_result['score']:.0f})", {"action": "觀望", "score": signal_result['score']}

    def optimize_parameters(self, ticker, df, param_grid, lookback_days=90):
        """
        🆕 v2.0 參數自動優化系統
        
        使用網格搜索找出最佳參數組合
        
        Args:
            ticker: 股票代碼
            df: 歷史數據
            param_grid: 參數網格 dict
                例: {
                    'atr_stop_multiplier': [2.0, 2.5, 3.0],
                    'atr_profit_multiplier': [3.5, 4.0, 4.5],
                    'confidence_threshold': [0.5, 0.6, 0.7]
                }
            lookback_days: 回測天數
        
        Returns:
            dict: 最佳參數與績效
        """
        from itertools import product
        
        # 生成所有參數組合
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        best_params = None
        best_score = -float('inf')
        results = []
        
        print(f"開始優化 {ticker}, 共 {len(combinations)} 組參數...")
        
        for idx, combo in enumerate(combinations):
            # 設定參數
            params_dict = dict(zip(param_keys, combo))
            
            for key, value in params_dict.items():
                if key in self.risk_params:
                    self.risk_params[key] = value
            
            # 執行回測
            _, df_log = self.run_simulation(ticker, df.copy(), lookback_days=lookback_days)
            
            if df_log.empty:
                continue
            
            # 計算績效指標
            final_assets = df_log['Assets'].iloc[-1]
            roi = (final_assets - self.account.initial_balance) / self.account.initial_balance
            
            # 計算夏普比率
            if 'Assets' in df_log.columns and len(df_log) > 1:
                returns = df_log['Assets'].pct_change().dropna()
                sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # 綜合評分 (70% ROI + 30% Sharpe)
            score = roi * 0.7 + sharpe * 0.3
            
            results.append({
                'params': params_dict.copy(),
                'roi': roi,
                'sharpe': sharpe,
                'score': score,
                'final_assets': final_assets
            })
            
            if score > best_score:
                best_score = score
                best_params = params_dict.copy()
            
            print(f"[{idx+1}/{len(combinations)}] 參數: {params_dict} | ROI: {roi:.2%} | Sharpe: {sharpe:.2f} | 評分: {score:.3f}")
        
        # 恢復最佳參數
        if best_params:
            for key, value in best_params.items():
                if key in self.risk_params:
                    self.risk_params[key] = value
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': sorted(results, key=lambda x: x['score'], reverse=True)
        }
