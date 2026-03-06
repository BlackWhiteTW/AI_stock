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
        stock_name = stock_info.get('longName', symbol)
        st.sidebar.metric(f"{stock_name} 現價", f"{current_price}")

        # 4. AI 智慧快報
        with st.sidebar.expander("🔍 AI 智慧快報", expanded=True):
            _render_ai_brief(symbol, ai_advisor)

        # 5. 下單面板子模組
        render_trade_panel(account, symbol, current_price)
        
        return symbol, hist_data, current_price, stock_info, df_stocks
        
    except Exception as e:
        st.sidebar.error(f"數據載入失敗: {e}")
        return input_code, pd.DataFrame(), 0, {}, df_stocks

def _render_ai_brief(symbol, ai_advisor):
    """內部渲染 AI 快報摘要"""
    try:
        summary = search_optimizer.get_cached_summary(symbol, ai_advisor)
        if summary:
            st.info(summary['ai_verdict'])
            st.caption(f"📊 情緒分數: {summary['sentiment']['score']:.1f}")
    except:
        st.write("快報載入中...")
