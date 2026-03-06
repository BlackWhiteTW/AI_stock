import streamlit as st
from utils import get_tw_stock_list
from services.financial_data import get_financial_manager
from .financial.sub_ratios import render_financial_ratios
from .financial.sub_statements import render_detailed_statements

def render_financial_page():
    """
    📊 財務報表分析 (彙整入口)
    負責股票選擇與分頁組織
    """
    st.title("📊 財務報表分析")
    
    # 1. 股票選擇器 (共用邏輯)
    stock_df = get_tw_stock_list()
    default_ticker = st.session_state.get('selected_ticker', '2330.TW')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_display = st.selectbox("選擇股票", stock_df['Display'].tolist(), index=0)
        ticker = stock_df[stock_df['Display'] == selected_display]['Ticker'].values[0]
        st.session_state['selected_ticker'] = ticker
    
    with col2:
        force_update = st.button("🔄 強制更新")

    st.divider()

    # 2. 獲取財務摘要數據
    financial_mgr = get_financial_manager()
    summary = financial_mgr.get_financial_summary(ticker)
    
    if summary.get('status') == 'no_data':
        st.error(f"❌ {summary.get('message', '無法獲取財務數據')}")
        return

    # 3. 組織子功能分頁
    tab_ratios, tab_details = st.tabs(["📈 核心指標分析", "📋 詳細財務報表"])
    
    with tab_ratios:
        render_financial_ratios(summary)
        
    with tab_details:
        render_detailed_statements(ticker, financial_mgr, force_update)
