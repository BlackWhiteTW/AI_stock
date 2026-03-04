import pandas as pd
import numpy as np
from services.financial_data import get_financial_manager
from services.performance_optimizer import get_indicator_cache, StreamlitCacheHelper

DEFAULT_PARAMS = {
    "rsi_lower": 30,
    "rsi_upper": 70,
    "rsi_window": 14,
    "ma_short": 20,
    "ma_long": 60
}

def add_financial_features(df, ticker):
    """
    添加財務報表特徵到技術分析數據中
    """
    try:
        financial_mgr = get_financial_manager()
        features = financial_mgr.get_financial_features(ticker)
        
        for feature_name, value in features.items():
            df[f'fin_{feature_name}'] = value
        return df
    except Exception as e:
        # 如果無法獲取財務數據，添加默認中性值
        print(f"無法添加財務特徵 [{ticker}]: {e}")
        default_features = {
            'financial_health_score': 50,
            'roe_normalized': 0,
            'debt_ratio_normalized': 50,
            'current_ratio_normalized': 50,
            'pe_ratio_normalized': 50,
            'dividend_yield_normalized': 0
        }
        for feature_name, value in default_features.items():
            df[f'fin_{feature_name}'] = value
        return df

def calculate_technical_indicators(df, params=None, use_cache=True, ticker=None, include_financial=True):
    """
    計算技術指標 (優化版 + 財務特徵整合 + Alpha 特徵)
    """
    if df.empty or len(df) < 30:
        return df
        
    # 優化 1: 檢查特徵是否已存在
    required_cols = ['RSI', 'MACD', 'MA20', 'Lower_Band', 'K', 'D', 'OBV', 'ATR', 'BIAS_20', 'Price_Velocity']
    if all(col in df.columns for col in required_cols):
        if include_financial and ticker and 'fin_financial_health_score' not in df.columns:
            df = add_financial_features(df, ticker)
        return df
    
    if params is None:
        params = DEFAULT_PARAMS
    
    # 優化 2: 使用快取系統
    if use_cache:
        try:
            cache = get_indicator_cache()
            data_hash = StreamlitCacheHelper.hash_dataframe(df)
            ticker_key = ticker if ticker else None
            cached_df = cache.get(ticker_key, params, data_hash)
            if cached_df is not None and not cached_df.empty:
                return cached_df
        except: pass
        
    rsi_window = params.get('rsi_window', 14)
    ma_short = params.get('ma_short', 20)
    ma_long = params.get('ma_long', 60)
    
    # --- 基礎指標 ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss.replace(0, 0.001)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['MA20'] = df['Close'].rolling(window=ma_short).mean() 
    df['StdDev'] = df['Close'].rolling(window=ma_short).std()
    df['Upper_Band'] = df['MA20'] + (df['StdDev'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['StdDev'] * 2)
    
    df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
    df['BIAS_20'] = ((df['Close'] - df['MA20']) / df['MA20'].replace(0, 0.001)) * 100
    
    # --- 進階指標 ---
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min).replace(0, 0.001) * 100
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ATR
    h_l = df['High'] - df['Low']
    h_pc = np.abs(df['High'] - df['Close'].shift())
    l_pc = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(14).mean()
    
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['MOM'] = df['Close'] - df['Close'].shift(10)
    
    # --- 🚀 Alpha 特徵 (2026 高級優化版) ---
    df['Price_Velocity'] = df['Close'].pct_change(periods=3)
    df['Price_Acceleration'] = df['Price_Velocity'].diff()
    df['ATR_Ratio'] = df['ATR'] / df['Close'].replace(0, 0.001)
    
    # 相對大盤強度
    if 'Market_Close' in df.columns:
        df['Relative_Strength'] = df['Close'].pct_change(10) - df['Market_Close'].pct_change(10)
    else:
        df['Relative_Strength'] = 0

    df['Dist_MA20'] = (df['Close'] - df['MA20']) / df['MA20'].replace(0, 0.001)
    df['Dist_MA60'] = (df['Close'] - df['MA_Long']) / df['MA_Long'].replace(0, 0.001)
    
    # 存入快取
    if use_cache:
        try:
            cache = get_indicator_cache()
            ticker_key = ticker if ticker else None
            data_hash = StreamlitCacheHelper.hash_dataframe(df)
            cache.set(ticker_key, params, df.copy(), data_hash)
        except: pass
    
    if include_financial and ticker:
        df = add_financial_features(df, ticker)

    return df
