import pandas as pd
import numpy as np

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
    
    :param df: 包含技術指標的 DataFrame
    :param ticker: 股票代碼
    :return: 添加財務特徵後的 DataFrame
    """
    try:
        from services.financial_data import get_financial_manager
        
        # 獲取財務特徵
        financial_mgr = get_financial_manager()
        features = financial_mgr.get_financial_features(ticker)
        
        # 將財務特徵作為常數列添加到 DataFrame
        # （因為財務報表是季度更新，在時間序列中保持不變）
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
    計算技術指標 (優化版 + 財務特徵整合)
    :param df: 股票數據 DataFrame
    :param params: 計算參數
    :param use_cache: 是否使用快取
    :param ticker: 股票代碼（用於添加財務特徵）
    :param include_financial: 是否包含財務特徵
    :return: 包含技術指標和財務特徵的 DataFrame
    """
    if df.empty or len(df) < 30:
        return df
        
    # 優化 1: 如果主要的技術指標欄位都已經存在，就不用重新計算
    required_cols = ['RSI', 'MACD', 'MA20', 'Lower_Band', 'K', 'D', 'OBV', 'ATR', 'BIAS_20']
    if all(col in df.columns for col in required_cols):
        # 如果需要財務特徵且尚未添加，則添加
        if include_financial and ticker and 'fin_financial_health_score' not in df.columns:
            df = add_financial_features(df, ticker)
        return df
    
    if params is None:
        params = DEFAULT_PARAMS
    
    # 優化 2: 使用快取系統
    # 🎯 有 ticker: 使用股票專用快取
    # 🔧 無 ticker: 使用通用快取 (用於回測、策略優化等)
    if use_cache:
        try:
            from services.performance_optimizer import get_indicator_cache, StreamlitCacheHelper
            cache = get_indicator_cache()
            
            # 生成數據哈希
            data_hash = StreamlitCacheHelper.hash_dataframe(df)
            
            # 嘗試從快取獲取
            ticker_key = ticker if ticker else None
            cached_df = cache.get(ticker_key, params, data_hash)
            if cached_df is not None and not cached_df.empty:
                return cached_df
        except:
            pass  # 快取失敗不影響主流程
        
    rsi_window = params.get('rsi_window', 14)
    ma_short = params.get('ma_short', 20)
    ma_long = params.get('ma_long', 60)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=ma_short).mean() 
    df['StdDev'] = df['Close'].rolling(window=ma_short).std()
    df['Upper_Band'] = df['MA20'] + (df['StdDev'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['StdDev'] * 2)
    
    # Long MA & BIAS (乖離率)
    df['MA_Long'] = df['Close'].rolling(window=ma_long).mean()
    df['BIAS_20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
    
    # --- 新增更豐富的訓練特徵 (2026.02 Update) ---
    
    # 1. KD (Stochastic Oscillator)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    
    # 2. OBV (On-Balance Volume) - 能量潮
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 3. ATR (Average True Range) - 真實波動幅度
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 4. Dividends (簡易殖利率代理變數 - 若有資料)
    if 'Dividends' in df.columns:
        # 計算近一年的累積配息與當前股價的比率 (Yield)
        df['Dividend_Sum_1Y'] = df['Dividends'].rolling(window=252).sum() # 約一年交易日
        df['Yield'] = df['Dividend_Sum_1Y'] / df['Close']
    else:
        df['Yield'] = 0
        
    # 5. Volatility (歷史波動率)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 6. Momentum (動能指標) - 10日變動
    df['MOM'] = df['Close'] - df['Close'].shift(10)

    # 7. CCI (商品路徑指標)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(window=14).mean()
    tp_mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad)
    
    # 優化 3: 計算完成後保存到快取
    # 🎯 有 ticker: 保存為股票專用快取
    # 🔧 無 ticker: 保存為通用快取
    if use_cache:
        try:
            from services.performance_optimizer import get_indicator_cache, StreamlitCacheHelper
            cache = get_indicator_cache()
            ticker_key = ticker if ticker else None
            data_hash = StreamlitCacheHelper.hash_dataframe(df)
            cache.set(ticker_key, params, df.copy(), data_hash)
        except:
            pass
    
    # 🆕 添加財務特徵（如果提供了 ticker）
    if include_financial and ticker:
        df = add_financial_features(df, ticker)

    return df
