import streamlit as st
import pandas as pd
import os
import time
import yfinance as yf
from services.market_data import MarketDataManager

# 初始化數據管理器
data_manager = MarketDataManager()

@st.cache_data(ttl=3600*12, show_spinner=False) # 12小時快取
def get_tw_stock_list():
    """獲取台股清單 (優化版 - 添加快取)"""
    csv_path = "data/tw_stock_list.csv"
    load_from_file = False

    # 檢查檔案是否存在且是今天的
    if os.path.exists(csv_path):
        try:
            mtime = os.path.getmtime(csv_path)
            file_date = time.strftime('%Y-%m-%d', time.localtime(mtime))
            today_date = time.strftime('%Y-%m-%d', time.localtime())
            
            # 如果是今天的檔案，就直接讀取
            if file_date == today_date:
                load_from_file = True
        except:
            pass

    if load_from_file:
        try:
            df = pd.read_csv(csv_path)
            # 確保必要的欄位存在，若檔案損壞則重新下載
            if 'Ticker' in df.columns and '產業' in df.columns:
                # 自動修復：如果發現檔案中沒有 "ETF" 分類 (代表是舊格式)，強制重新下載
                if 'ETF' not in df['產業'].unique():
                    load_from_file = False
                else:
                    return df
        except Exception as e:
            pass
    
    # 如果沒有檔案或需要更新，則從網路下載
    try:
        # 用戶指定：只記錄「上市」股票 (Mode=2)
        url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        df_twse = pd.read_html(url_twse, encoding='cp950')[0]
        
        # 整理數據函數
        def clean_df(df, suffix):
            if df.shape[1] >= 7:
                df = df.iloc[:, :7]
            df.columns = ["有價證券代號及名稱", "ISIN", "上市日", "市場", "產業", "CFI", "備註"]
            df = df.dropna(subset=['有價證券代號及名稱', 'ISIN'])
            
            # --- 優化：過濾掉權證 (Warrants) 與非股票商品 ---
            # 許多 03xxxx, 04xxxx 等都是權證，yfinance 無法抓取
            # 利用 CFI 代碼篩選：保留 'E'(股票) 與 'C'(ETF/基金) 開頭
            if 'CFI' in df.columns:
                # 識別 ETF:
                # 1. CFI 以 'C' 開頭 (標準)
                # 2. 代號以 '00' 開頭 (常見台股 ETF, 如 0050, 00878, 00919)
                is_cfi_etf = df['CFI'].astype(str).str.startswith('C', na=False)
                is_code_etf = df['有價證券代號及名稱'].astype(str).str.strip().str.startswith('00')
                
                is_etf = is_cfi_etf | is_code_etf
                
                # 強制將這些項目的產業設為 "ETF"
                df.loc[is_etf, '產業'] = 'ETF'
                
                # 進行篩選: 保留股票(E), ETF(C), 或已被我們標記為 ETF 的項目
                df = df[
                    df['CFI'].astype(str).str.startswith(('E', 'C'), na=False) | 
                    (df['產業'] == 'ETF')
                ]

            df['產業'] = df['產業'].fillna('其他')
            
            split_df = df['有價證券代號及名稱'].str.split(r'[　\s]+', n=1, expand=True)
            if split_df.shape[1] >= 2:
                df['代碼'] = split_df[0]
                df['名稱'] = split_df[1]
            else:
                return pd.DataFrame() 
            
            df = df[df['代碼'].str.match(r'^[0-9A-Z]{4,6}$', na=False)]
            df['Ticker'] = df['代碼'] + suffix
            return df[['代碼', '名稱', 'Ticker', '產業', '市場']]

        full_df = clean_df(df_twse, ".TW")
        
        # 建構顯示欄位
        full_df = full_df.dropna(subset=['代碼', '名稱', '產業'])
        full_df['Display'] = full_df['代碼'] + " " + full_df['名稱']
        
        # 存檔 (只存上市)
        full_df.to_csv(csv_path, index=False)
        
        return full_df
    except Exception as e:
        print(f"Error fetching list: {e}")
        # 如果網路失敗，嘗試讀取舊檔案
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

def get_stock_data(code, force_offline=False, skip_info=False):
    # 如果使用者沒有輸入 .TW 或 .TWO，嘗試自動補全
    ticker_candidates = [code]
    if not (code.endswith('.TW') or code.endswith('.TWO')):
        ticker_candidates = [f"{code}.TW", f"{code}.TWO"] 
    else:
        ticker_candidates = [code]

    final_ticker = code
    hist = pd.DataFrame()
    info = {}
    
    # 增加隨機延遲，避免多執行緒下觸發 Rate Limit
    if not force_offline:
        import time
        import random
        time.sleep(random.uniform(0.1, 0.5))

    for t in ticker_candidates:
        # 使用 DataManager 讀取與自動更新
        # load_data 會自動檢查本地 -> 比較日期 -> 下載新數據 -> 存檔
        temp_hist = data_manager.load_data(t, only_cached=force_offline)
        
        if not temp_hist.empty:
            final_ticker = t
            hist = temp_hist
            if not skip_info and not force_offline:
                try:
                    info = yf.Ticker(t).info
                except:
                    info = {}
            break
            
    if hist.empty:
        # 如果都找不到，回傳空
        return code, pd.DataFrame(), 0, {}

    # yfinance 有時 info 會缺漏，做個 fallback
    current_price = 0
    if not skip_info:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or (hist['Close'].iloc[-1] if not hist.empty else 0)
    
    # 回傳完整的 hist
    return final_ticker, hist, current_price, info

def get_stock_with_industry_news(code, force_offline=False):
    """
    獲取股票數據並自動查詢產業新聞（增強版）
    
    Args:
        code: 股票代碼
        force_offline: 是否強制離線模式
    
    Returns:
        {
            'ticker': 股票代碼,
            'hist': 歷史數據,
            'current_price': 當前價格,
            'info': 股票資訊,
            'industry': 所屬產業,
            'industry_news': 產業新聞,
            'industry_sentiment': 產業情緒分數,
            'industry_summary': 產業情緒摘要
        }
    """
    # 獲取基本股票數據
    ticker, hist, current_price, info = get_stock_data(code, force_offline=force_offline)
    
    result = {
        'ticker': ticker,
        'hist': hist,
        'current_price': current_price,
        'info': info,
        'industry': None,
        'industry_news': [],
        'industry_sentiment': 0,
        'industry_summary': "無產業資訊"
    }
    
    if force_offline or hist.empty:
        return result
    
    try:
        from services.news_analyzer import NewsAnalyzer
        
        analyzer = NewsAnalyzer()
        stock_name = info.get('longName', '') or info.get('shortName', '')
        
        # 檢測產業
        industry = analyzer.detect_industry(ticker, stock_name)
        
        if industry:
            result['industry'] = industry
            
            # 獲取產業新聞和情緒分析
            score, summary, details = analyzer.analyze_industry_sentiment(industry, detailed=True)
            
            result['industry_sentiment'] = score
            result['industry_summary'] = summary
            result['industry_news'] = details.get('matched_news', [])
            result['industry_details'] = details
            
    except Exception as e:
        print(f"無法獲取產業新聞: {e}")
    
    return result
