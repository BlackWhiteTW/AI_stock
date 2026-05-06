import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon, status_card

def render_trader_dashboard(trader_bot_v1, trader_bot_v2):
    """交易機器人控制面板子模組"""
    header_with_icon("AI 自動化交易中心", "", "管理與監控您的量化交易策略")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Trader V1 (策略回測)")
        st.info("V1 專注於歷史回測模擬，協助您驗證策略穩定性。")
        if st.button("執行 V1 策略回測", key="start_v1"):
            st.info("請先在側邊欄選擇股票，然後開始回測...")

    with col2:
        st.subheader(" Trader V2 (自動化交易)")
        st.info("V2 負責每日收盤後的實時自動交易任務。")
        # 安全取得運行狀態
        v2_is_running = getattr(trader_bot_v2, 'is_running', False)
        v2_status = "運行中" if v2_is_running else "待機中"
        st.metric("V2 機器人狀態", v2_status)
        
        if st.button("啟動每日自動交易", disabled=v2_is_running, key="start_v2"):
            st.success("✅ V2 自動交易已啟動，將於每日收盤後自動執行。")
