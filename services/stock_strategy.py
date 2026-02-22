import pandas as pd
import numpy as np
from . import tech_analysis  # 確保同目錄下有 tech_analysis.py

class StockStrategy:
    # [修正重點] 加入 *args, **kwargs 以容錯，解決 "3 arguments were given" 報錯
    def __init__(self, default_params=None, *args, **kwargs):
        if default_params is None:
            # 嘗試讀取 tech_analysis 的預設參數，若無則使用空字典
            self.default_params = getattr(tech_analysis, 'DEFAULT_PARAMS', {})
        else:
            self.default_params = default_params
        self.learned_params = {}

    def get_params(self, ticker):
        return self.learned_params.get(ticker, self.default_params)

    def optimize_strategy_basic(self, ticker, df):
        """參數優化回測 (保留原有功能)"""
        best_score = -float('inf')
        best_params = self.default_params.copy()
        
        param_grid = [
            {"rsi_lower": 30, "rsi_upper": 70, "ma_short": 20, "ma_long": 60, "name": "標準均衡型"},
            {"rsi_lower": 20, "rsi_upper": 80, "ma_short": 20, "ma_long": 60, "name": "保守穩健型"},
            {"rsi_lower": 35, "rsi_upper": 65, "ma_short": 10, "ma_long": 40, "name": "積極短線型"},
            {"rsi_lower": 25, "rsi_upper": 75, "ma_short": 50, "ma_long": 120, "name": "長線投資型"}
        ]
        
        for p in param_grid:
            roi, _ = self._run_backtest(df.copy(), p)
            if roi > best_score:
                best_score = roi
                best_params = p
        
        self.learned_params[ticker] = best_params
        best_name = best_params.get('name', '自定義')
        return f"✅ 基礎策略優化完成：發現 **「{best_name}」** 參數最適合 (ROI: {best_score:.1f}%)"

    def _run_backtest(self, df, params):
        df = tech_analysis.calculate_technical_indicators(df, params)
        df = df.dropna()
        if df.empty: return -100, 0
        
        balance = 100000
        position = 0
        shares = 0
        
        rsi_lower = params['rsi_lower']
        rsi_upper = params['rsi_upper']
        
        trades = 0
        
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            # 這裡加上保護，避免 RSI 取值錯誤
            if i-1 < len(df):
                rsi = df['RSI'].iloc[i-1]
            else:
                continue
            
            if position == 0 and rsi < rsi_lower:
                shares = int(balance // price)
                balance -= shares * price
                position = 1
                trades += 1
            elif position == 1 and rsi > rsi_upper:
                balance += shares * price
                shares = 0
                position = 0
                trades += 1
                
        final_value = balance + (shares * df['Close'].iloc[-1])
        roi = ((final_value - 100000) / 100000) * 100
        return roi, trades
    
    def analyze_stock(self, ticker, df, ai_pred_price=None, news_analyzer=None, include_news=True):
        """
        綜合分析：基本參數 + 深度學習預測 (針對中長線與波段優化)
        """
        # [防呆] 確保數據足夠
        if df.empty or len(df) < 60:
            return {"action": "未知", "score": 0, "reasons": ["數據不足 (需至少60筆)"], "price": 0}

        params = self.get_params(ticker)
        df = tech_analysis.calculate_technical_indicators(df, params)
        
        last_close = df['Close'].iloc[-1]
        prev_close = current_vol = 0 
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
        
        # 指標取得 (加入安全預設值)
        rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df else 0
        ma_short = df['MA20'].iloc[-1] if 'MA20' in df else last_close
        ma_long = df['MA_Long'].iloc[-1] if 'MA_Long' in df else last_close
        
        # 計算 MA5
        ma5 = df['Close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else last_close
        
        # 成交量計算
        vol_ma20 = df['Volume'].rolling(window=20).mean().iloc[-1] if 'Volume' in df else 0
        current_vol = df['Volume'].iloc[-1] if 'Volume' in df else 0
        
        reasons = []
        score = 0
        
        # -------------------------------------------------------------
        # 0. 流動性濾網 (Liquidity Filter)
        # -------------------------------------------------------------
        if vol_ma20 < 200000: 
            return {
                "action": "⛔ 流動性不足", 
                "score": -10, 
                "reasons": [f"⚠️ 成交量過低 (月均量僅 {int(vol_ma20/1000)} 張)，風險高"], 
                "price": last_close
            }

        # -------------------------------------------------------------
        # 1. 嚴格趨勢判定 (Trend Filter)
        # -------------------------------------------------------------
        trend = "Neutral"
        is_bull_align = ma5 > ma_short > ma_long
        
        ma_long_prev = df['MA_Long'].iloc[-5] if len(df) >= 5 else ma_long
        is_ma_long_rising = ma_long > ma_long_prev
        
        if is_bull_align and is_ma_long_rising:
            trend = "Strong Bull"
            if last_close > ma5:
                score += 3
                reasons.append("📈 [核心] 均線完美多頭排列且季線翻揚")
            else:
                score += 2
                reasons.append("📈 [核心] 長線多頭架構，沿 5日線整理")
        elif ma_short > ma_long:
            if not is_ma_long_rising:
                score -= 1
                trend = "Fake Bull"
                reasons.append("⚠️ [核心] 假多頭：季線尚未翻揚")
            elif last_close > ma_long:
                trend = "Bullish Correction"
                reasons.append("📉 [核心] 多頭拉回整理")
            else:
                trend = "Weak"
                score -= 3
                reasons.append("⚠️ [核心] 轉弱：股價跌破季線")
        else:
            if last_close < ma_short:
                trend = "Bearish" 
                score -= 5 
                reasons.append("📉 [核心] 均線空頭排列，趨勢向下")
            else:
                trend = "Rebound"
                score -= 2
                reasons.append("🤕 [核心] 弱勢反彈")

        # -------------------------------------------------------------
        # 2. 動能與位階 (Momentum)
        # -------------------------------------------------------------
        if rsi > 80:
            score -= 3
            reasons.append(f"⚠️ RSI 過熱 ({rsi:.1f})")
        elif rsi > 70:
            if "Strong Bull" in trend:
                reasons.append(f"🔥 RSI 強勢區 ({rsi:.1f})")
            else:
                score -= 1 
                reasons.append(f"⚠️ 弱勢反彈 RSI 偏高")
        elif 50 <= rsi <= 70:
            if "Strong Bull" in trend:
                score += 1
                reasons.append("🚀 RSI 位於強勢區")
        elif rsi < 30:
            if "Bearish" not in trend:
                score += 1
                reasons.append(f"🟢 RSI 低檔 ({rsi:.1f}) 醞釀反彈")
            else:
                score -= 2 
                reasons.append(f"🔴 RSI 殺盤鈍化 ({rsi:.1f})")

        macd_hist = macd - df['Signal_Line'].iloc[-1] if 'Signal_Line' in df else 0
        if macd > 0 and macd_hist > 0:
            if "Bull" in trend:
                score += 1
                reasons.append("🐂 MACD 紅柱放大")
        
        # -------------------------------------------------------------
        # 3. 成交量 (Volume)
        # -------------------------------------------------------------
        price_change_pct = 0
        if prev_close > 0:
            price_change_pct = ((last_close - prev_close) / prev_close) * 100
        
        if vol_ma20 > 0:
            vol_ratio = current_vol / vol_ma20
            if vol_ratio > 1.5:
                if price_change_pct > 2:
                    score += 2
                    reasons.append(f"🔥 出量攻擊 (量增{int(vol_ratio*100-100)}%)")
                elif price_change_pct < -2:
                    score -= 3
                    reasons.append(f"💀 爆量長黑 (量增{int(vol_ratio*100-100)}%)")
            elif vol_ratio < 0.6:
                if "Correction" in trend:
                    score += 1
                    reasons.append("💧 拉回量縮籌碼穩定")

        # -------------------------------------------------------------
        # 4. AI 深度學習預測
        # -------------------------------------------------------------
        if ai_pred_price:
            pred_change = ((ai_pred_price - last_close) / last_close) * 100
            
            if pred_change < -1.5:
                score -= 4 
                reasons.insert(0, f"⚠️ AI 看空: 下看 {ai_pred_price:.2f} ({pred_change:.2f}%)")
            elif pred_change < -0.5:
                score -= 2
                reasons.insert(0, f"📉 AI 預測修正: {pred_change:.2f}%")
            elif pred_change > 1.5:
                if "Bearish" not in trend:
                    score += 2 
                    reasons.insert(0, f"🤖 AI 強力看多: 目標 {ai_pred_price:.2f} (+{pred_change:.2f}%)")
                else:
                    score -= 1 
                    reasons.append(f"🤖 AI 看多但趨勢偏空 (風險大)")
            elif pred_change > 0.5:
                 if "Bull" in trend:
                    score += 1
                    reasons.insert(0, f"📈 AI 預測緩漲 (+{pred_change:.2f}%)")
        else:
            reasons.append("⚪ (無 AI 模型)")
            
        # -------------------------------------------------------------
        # 5. 新聞面情緒
        # -------------------------------------------------------------
        if include_news and news_analyzer:
            try:
                news_score, news_details = news_analyzer.analyze_sentiment(ticker)
                if news_score > 0:
                    icon = "🔥" if news_score >= 2 else "📰"
                    reasons.insert(0, f"{icon} 新聞利多 (+{news_score}分)")
                    score += news_score
                    for i, detail in enumerate(news_details[:2]): # 只取前2則
                         reasons.append(f"   └ 消息: {detail}")
                elif news_score < 0:
                    icon = "⛈️" if news_score <= -2 else "📰"
                    reasons.insert(0, f"{icon} 新聞利空 ({news_score}分)")
                    score += news_score 
                    for i, detail in enumerate(news_details[:2]):
                         reasons.append(f"   └ 警示: {detail}")
            except Exception:
                pass

        # -------------------------------------------------------------
        # 6. 最終結算
        # -------------------------------------------------------------
        if score >= 7:
            action = "⭐️ 強力買進"
        elif score >= 5:
            action = "🟢 建議買進"
        elif score >= 2:
            action = "👀 觀察清單"
        elif score <= -5:
            action = "💀 強力賣出"
        elif score <= -2:
            action = "🔴 建議賣出"
        else:
            action = "⚪ 觀望 / 中立"
            
        # 強制歸零機制：季線下的股票不給高分
        if "Bearish" in trend and score > 0:
            score = 0 
            action = "⚠️ 觀望 (空頭走勢)"
            reasons.insert(0, "🛑 強制歸零：股價位於季線之下")

        return {
            "action": action,
            "score": score,
            "reasons": reasons,
            "price": last_close
        }