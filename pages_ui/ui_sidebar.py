import streamlit as st
import pandas as pd
from utils import get_tw_stock_list, get_stock_data
from services.search_optimizer import SearchOptimizer
from .sidebar.sub_search import render_stock_search
from .sidebar.sub_trade_panel import render_trade_panel

# 初始化搜尋優化器
search_optimizer = SearchOptimizer()


def render_sidebar(account, ai_advisor=None):
    st.sidebar.title("交易操作面板")

    # 1. 載入清單
    try:
        df_stocks = get_tw_stock_list()
    except:
        df_stocks = pd.DataFrame()

    # 2. 搜尋子模組
    input_code = render_stock_search(df_stocks, ai_advisor)

    # 3. 獲取核心數據
    try:
        symbol, hist_data, current_price, stock_info = get_stock_data(input_code)
        stock_name = stock_info.get("longName", symbol)
        st.sidebar.metric(f"{stock_name} 現價", f"{current_price}")

        # 4. AI 智慧快報 (暫時隱藏以確保系統流暢度)
        # with st.sidebar.expander(" AI 智慧快報", expanded=True):
        #     try:
        #         _render_ai_brief(symbol, ai_advisor)
        #     except Exception as brief_error:
        #         st.error(f"AI 快報錯誤: {brief_error}")

        # 5. 下單面板子模組
        render_trade_panel(account, symbol, current_price)

        return symbol, hist_data, current_price, stock_info, df_stocks

    except Exception as e:
        st.sidebar.error(f"數據載入失敗: {e}")
        return input_code, pd.DataFrame(), 0, {}, df_stocks


def _render_ai_brief(symbol, ai_advisor):
    """內部渲染 AI 快報摘要 (非阻塞與手動觸發版本)"""
    try:
        # 1. 優先嘗試從快取管理器直接讀取 (不觸發生成，避免阻塞)
        report = search_optimizer.cache_manager.get("analysis_report", symbol)
        
        if report:
            # 安全地訪問欄位
            ai_verdict = report.get("ai_verdict", "暫無分析內容")
            sentiment = report.get("sentiment", {})
            score = sentiment.get("score", 0) if isinstance(sentiment, dict) else 0
            ts = report.get("timestamp", "未知時間")
            
            st.info(ai_verdict)
            st.caption(f" 情緒分數: {score:.1f} |  {ts}")
            
            # 提供手動更新按鈕
            if st.button(" 更新分析", key=f"btn_update_{symbol}"):
                with st.spinner("正在重新分析中..."):
                    search_optimizer.get_smart_summary(symbol, ai_advisor, force_update=True)
                    st.rerun()
        else:
            # 若無報告，不自動生成，改為手動按鈕觸發
            st.warning("⚡ 尚無此股票的 AI 分析報告")
            if st.button(" 生成 AI 智慧快報", key=f"btn_gen_{symbol}"):
                with st.spinner("聯網搜尋並分析中 (約 10-15s)..."):
                    search_optimizer.get_smart_summary(symbol, ai_advisor, force_update=True)
                    st.rerun()

    except Exception as e:
        st.error(f"⚠️ 快報處理異常: {str(e)[:50]}")
