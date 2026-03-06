import streamlit as st
from .testing.sub_api_test import render_api_testing

def render_tab_testing():
    """
    🧪 AI 實驗室 & 診斷 (彙整入口)
    """
    st.header("🧪 AI 實驗室 & 系統診斷")
    st.caption("開發者與進階用戶專區：檢查系統穩定性與模型狀態")
    
    test_tab1, test_tab2 = st.tabs(["🌐 服務連線測試", "🧠 模型結構診斷"])
    
    with test_tab1:
        render_api_testing()
        
    with test_tab2:
        st.info("模型結構診斷模組開發中... 未來將提供神經網路層級的可視化。")
        st.code("""
        Current Architecture: HybridTransLSTM
        - Transformer Layers: 6
        - LSTM Layers: 3
        - Attention Heads: 16
        - Hidden Dim: 256
        """)
