"""
🌍 市場情緒分析模組 v1.0
提供大盤情緒、資金流向、異常檢測等市場級別指標
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

class MarketSentimentAnalyzer:
    """市場情緒分析器"""
    
    def __init__(self, cache_duration=3600):
        """
        :param cache_duration: 快取持續時間（秒），預設1小時
        """
        self.cache_file = "data/market_sentiment_cache.json"
        self.cache_duration = cache_duration
        self.market_index = "^TWII"  # 台灣加權指數
        
    def get_market_sentiment(self):
        """
        獲取大盤情緒（貪婪/恐懼/中性）
        :return: dict {'sentiment': str, 'score': float, 'rsi': float}
        """
        # 檢查快取
        cache = self._load_cache()
        if cache and 'market_sentiment' in cache:
            cache_time = datetime.fromisoformat(cache['timestamp'])
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                return cache['market_sentiment']
        
        try:
            # 下載大盤近3個月數據
            twii = yf.download(self.market_index, period='3mo', progress=False)
            
            # 安全檢查：確保數據有效且可用
            if twii is None or not isinstance(twii, pd.DataFrame):
                return {'sentiment': 'Unknown', 'score': 50, 'rsi': 50}
            
            if len(twii) == 0:
                return {'sentiment': 'Unknown', 'score': 50, 'rsi': 50}
            
            # 檢查是否有 Close 列
            if 'Close' not in twii.columns:
                return {'sentiment': 'Unknown', 'score': 50, 'rsi': 50}
            
            # 計算大盤 RSI
            rsi = self._calculate_rsi(twii['Close'])
            
            # 判斷情緒
            if rsi > 70:
                sentiment = 'Greedy'  # 貪婪（過熱）
                score = min(100, rsi)
            elif rsi < 30:
                sentiment = 'Fear'    # 恐懼（超跌）
                score = max(0, rsi)
            else:
                sentiment = 'Neutral' # 中性
                score = rsi
            
            result = {
                'sentiment': sentiment,
                'score': score,
                'rsi': rsi,
                'market_index': self.market_index
            }
            
            # 更新快取
            self._save_to_cache('market_sentiment', result)
            return result
            
        except Exception as e:
            print(f"⚠️ 獲取大盤情緒失敗: {e}")
            import traceback
            traceback.print_exc()
            return {'sentiment': 'Unknown', 'score': 50, 'rsi': 50}
    
    def detect_volume_anomaly(self, df, threshold=2.0):
        """
        檢測異常成交量
        :param df: 股票數據 DataFrame
        :param threshold: 異常倍數（預設2倍）
        :return: pandas Series (True/False)
        """
        if df is None or len(df) == 0 or len(df) < 20:
            return pd.Series([False] * len(df))
        
        # 計算20日平均成交量
        avg_volume = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # 成交量比率
        volume_ratio = df['Volume'] / avg_volume
        
        # 異常標記（超過閾值倍數）
        is_anomaly = volume_ratio > threshold
        
        return is_anomaly
    
    def calculate_money_flow_index(self, df, period=14):
        """
        計算資金流向指標 (MFI)
        :param df: 股票數據 DataFrame
        :param period: 計算週期
        :return: pandas Series
        """
        if df is None or len(df) == 0 or len(df) < period:
            return pd.Series([50] * len(df))
        
        # 典型價格
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # 資金流量 = 典型價格 × 成交量
        money_flow = typical_price * df['Volume']
        
        # 價格變動方向
        price_change = typical_price.diff()
        
        # 正/負資金流
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        # 計算 MFI
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        return mfi.fillna(50)
    
    def calculate_adx(self, df, period=14):
        """
        計算平均趨向指標 (ADX) - 判斷趨勢強度
        :param df: 股票數據 DataFrame
        :param period: 計算週期
        :return: pandas Series
        """
        if df is None or len(df) == 0 or len(df) < period * 2:
            return pd.Series([25] * len(df))
        
        # 計算 +DM 和 -DM
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # ATR (已在 tech_analysis 計算，這裡簡化版)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # +DI 和 -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX (DX 的平滑平均)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(25)
    
    def get_price_position(self, df, period=252):
        """
        計算股價位置（相對於52週高低點）
        :param df: 股票數據 DataFrame
        :param period: 週期（預設252個交易日≈1年）
        :return: pandas Series (0-100)
        """
        if df is None or len(df) == 0:
            return pd.Series([50] * len(df) if len(df) > 0 else [50])
        if len(df) < period:
            period = len(df)
        
        # 計算期間內的最高價和最低價
        highest = df['High'].rolling(window=period, min_periods=1).max()
        lowest = df['Low'].rolling(window=period, min_periods=1).min()
        
        # 當前價格位置 (0-100)
        price_range = highest - lowest
        price_position = ((df['Close'] - lowest) / price_range * 100).fillna(50)
        
        return price_position
    
    def detect_sharp_change(self, df, threshold=0.05):
        """
        檢測急漲急跌（單日漲跌超過閾值）
        :param df: 股票數據 DataFrame
        :param threshold: 閾值（預設5%）
        :return: dict {'sharp_up': Series, 'sharp_down': Series}
        """
        if df is None or len(df) == 0:
            df_len = len(df) if df is not None else 0
            return {
                'sharp_up': pd.Series([False] * df_len),
                'sharp_down': pd.Series([False] * df_len)
            }
        
        # 計算日報酬率
        daily_return = df['Close'].pct_change()
        
        # 急漲/急跌標記
        sharp_up = daily_return > threshold
        sharp_down = daily_return < -threshold
        
        return {
            'sharp_up': sharp_up,
            'sharp_down': sharp_down
        }
    
    def get_multi_timeframe_trend(self, ticker):
        """
        獲取多時間框架趨勢（日線、週線、月線）
        :param ticker: 股票代碼
        :return: dict
        """
        try:
            # 下載較長期數據
            stock = yf.download(ticker, period='2y', progress=False)
            
            # 安全檢查
            if stock is None or not isinstance(stock, pd.DataFrame):
                return {'daily': 'Unknown', 'weekly': 'Unknown', 'monthly': 'Unknown'}
            
            if len(stock) == 0:
                return {'daily': 'Unknown', 'weekly': 'Unknown', 'monthly': 'Unknown'}
            
            # 日線趨勢
            daily_trend = self._get_trend(stock, ma_short=20, ma_long=60)
            
            # 週線趨勢（重採樣）
            weekly = stock.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            weekly_trend = self._get_trend(weekly, ma_short=10, ma_long=30)
            
            # 月線趨勢
            monthly = stock.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            monthly_trend = self._get_trend(monthly, ma_short=6, ma_long=12)
            
            return {
                'daily': daily_trend,
                'weekly': weekly_trend,
                'monthly': monthly_trend
            }
            
        except Exception as e:
            print(f"⚠️ 多時間框架分析失敗: {e}")
            return {'daily': 'Unknown', 'weekly': 'Unknown', 'monthly': 'Unknown'}
    
    def _get_trend(self, df, ma_short=20, ma_long=60):
        """判斷趨勢方向"""
        if df is None or len(df) == 0 or len(df) < ma_long:
            return 'Unknown'
        
        ma_s = df['Close'].rolling(ma_short).mean().iloc[-1]
        ma_l = df['Close'].rolling(ma_long).mean().iloc[-1]
        price = df['Close'].iloc[-1]
        
        if price > ma_s > ma_l:
            return 'Uptrend'    # 上升趨勢
        elif price < ma_s < ma_l:
            return 'Downtrend'  # 下降趨勢
        else:
            return 'Sideways'   # 盤整
    
    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        if prices is None or len(prices) == 0:
            return 50.0
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # 安全處理除以零的情況
            loss_safe = loss.copy()
            loss_safe[loss_safe == 0] = 1e-10
            
            rs = gain / loss_safe
            rsi = 100 - (100 / (1 + rs))
            
            # 確保返回單一數值
            if isinstance(rsi, pd.Series):
                return float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
            return float(rsi) if not pd.isna(rsi) else 50.0
        except Exception as e:
            return 50.0
    
    def _load_cache(self):
        """載入快取"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, key, value):
        """儲存到快取"""
        cache = self._load_cache() or {}
        cache[key] = value
        cache['timestamp'] = datetime.now().isoformat()
        
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


# 快速測試
if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    
    # 測試大盤情緒
    sentiment = analyzer.get_market_sentiment()
    print(f"📊 大盤情緒: {sentiment}")
    
    # 測試多時間框架
    trends = analyzer.get_multi_timeframe_trend('2330.TW')
    print(f"📈 多時間框架趨勢: {trends}")
