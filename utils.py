import streamlit as st
import pandas as pd
import os
import time
import random
import yfinance as yf
from services.market_data import MarketDataManager
from services.news_analyzer import NewsAnalyzer

# 初始化數據管理器
data_manager = MarketDataManager()


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def get_tw_stock_list():
    """獲取台股清單 (優化版 - 添加快取與診斷日誌)"""
    print("📋 [Data] 正在獲取台股清單...")
    csv_path = "data/tw_stock_list.csv"
    load_from_file = False

    # 檢查檔案是否存在且是今天的
    if os.path.exists(csv_path):
        try:
            mtime = os.path.getmtime(csv_path)
            file_date = time.strftime("%Y-%m-%d", time.localtime(mtime))
            today_date = time.strftime("%Y-%m-%d", time.localtime())

            if file_date == today_date:
                print(f"📄 [Data] 發現今日快取檔案: {csv_path}")
                load_from_file = True
        except Exception as e:
            print(f"⚠️ [Data] 檢查快取檔案狀態失敗: {e}")

    if load_from_file:
        try:
            df = pd.read_csv(csv_path)
            if "Ticker" in df.columns and "產業" in df.columns:
                if "ETF" not in df["產業"].unique():
                    print("⚠️ [Data] 快取檔案格式過舊，準備重新下載...")
                    load_from_file = False
                else:
                    print(f"✅ [Data] 成功從本地載入 {len(df)} 檔股票資訊")
                    return df
        except Exception as e:
            print(f"⚠️ [Data] 讀取本地 CSV 失敗: {e}")

    # 從網路下載
    print("🌐 [Data] 正在從證交所下載最新清單 (這可能需要幾秒鐘)...")
    try:
        url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        # 加入超時控制 (藉由 requests 獲取內容後再交給 pandas)
        import requests

        response = requests.get(url_twse, timeout=10)
        df_twse = pd.read_html(response.text, encoding="cp950")[0]
        print("✅ [Data] 證交所數據下載成功，正在整理格式...")

        # 整理數據函數
        def clean_df(df, suffix):
            if df.shape[1] >= 7:
                df = df.iloc[:, :7]
            df.columns = [
                "有價證券代號及名稱",
                "ISIN",
                "上市日",
                "市場",
                "產業",
                "CFI",
                "備註",
            ]
            df = df.dropna(subset=["有價證券代號及名稱", "ISIN"])

            if "CFI" in df.columns:
                is_cfi_etf = df["CFI"].astype(str).str.startswith("C", na=False)
                is_code_etf = (
                    df["有價證券代號及名稱"]
                    .astype(str)
                    .str.strip()
                    .str.startswith("00")
                )
                is_etf = is_cfi_etf | is_code_etf
                df.loc[is_etf, "產業"] = "ETF"
                df = df[
                    df["CFI"].astype(str).str.startswith(("E", "C"), na=False)
                    | (df["產業"] == "ETF")
                ]

            df["產業"] = df["產業"].fillna("其他")
            split_df = df["有價證券代號及名稱"].str.split(r"[　\s]+", n=1, expand=True)
            if split_df.shape[1] >= 2:
                df["代碼"] = split_df[0]
                df["名稱"] = split_df[1]
            else:
                return pd.DataFrame()

            df = df[df["代碼"].str.match(r"^[0-9A-Z]{4,6}$", na=False)]
            df["Ticker"] = df["代碼"] + suffix
            return df[["代碼", "名稱", "Ticker", "產業", "市場"]]

        full_df = clean_df(df_twse, ".TW")
        full_df = full_df.dropna(subset=["代碼", "名稱", "產業"])
        full_df["Display"] = full_df["代碼"] + " " + full_df["名稱"]

        os.makedirs("data", exist_ok=True)
        full_df.to_csv(csv_path, index=False)
        print(f"✅ [Data] 清單更新完成，共 {len(full_df)} 檔")
        return full_df
    except Exception as e:
        print(f"❌ [Data] 獲取網路清單失敗: {e}")
        if os.path.exists(csv_path):
            print("🔄 [Data] 嘗試讀取舊有本地檔案作為備案...")
            return pd.read_csv(csv_path)
        return pd.DataFrame()


def get_stock_data(code, force_offline=False, skip_info=False, use_cache=True):
    """獲取單股數據與資訊 (保證不卡死版本)"""
    print(f"📥 [Data] 請求數據: {code} (use_cache={use_cache})")
    ticker_candidates = [code]
    if not (code.endswith(".TW") or code.endswith(".TWO")):
        ticker_candidates = [f"{code}.TW", f"{code}.TWO"]

    if force_offline:
        return code, pd.DataFrame(), 0, {}

    for t in ticker_candidates:
        try:
            # 讓資料管理器自行判斷快取是否需要補抓最新資料
            df = data_manager.load_data(t, only_cached=False)

            if df is not None and not df.empty:
                current_price = df["Close"].iloc[-1]
                return t, df, current_price, {"shortName": t}
        except Exception as e:
            print(f"⚠️ [Data] 獲取 {t} 失敗: {e}")

    return code, pd.DataFrame(), 0, {}


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
    ticker, hist, current_price, info = get_stock_data(
        code, force_offline=force_offline
    )

    result = {
        "ticker": ticker,
        "hist": hist,
        "current_price": current_price,
        "info": info,
        "industry": None,
        "industry_news": [],
        "industry_sentiment": 0,
        "industry_summary": "無產業資訊",
    }

    if force_offline or hist.empty:
        return result

    try:
        from services.news_analyzer import NewsAnalyzer

        analyzer = NewsAnalyzer()
        stock_name = info.get("longName", "") or info.get("shortName", "")

        # 檢測產業
        industry = analyzer.detect_industry(ticker, stock_name)

        if industry:
            result["industry"] = industry

            # 獲取產業新聞和情緒分析
            score, summary, details = analyzer.analyze_industry_sentiment(
                industry, detailed=True
            )

            result["industry_sentiment"] = score
            result["industry_summary"] = summary
            result["industry_news"] = details.get("matched_news", [])
            result["industry_details"] = details

    except Exception as e:
        print(f"無法獲取產業新聞: {e}")

    return result
