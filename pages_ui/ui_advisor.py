import streamlit as st
import os
import json
import datetime
import traceback
import pandas as pd
from .advisor.sub_analysis import render_stock_analysis
from .advisor.sub_chat import render_ai_chat

def render_tab_advisor(ai_advisor, symbol, hist_data, current_price, account, df_stocks):
    """
    🤖 AI 智能投資顧問 (彙整入口)
    負責將分析、對話與預測驗證模組組合成完整的分頁介面
    """
    st.header("🤖 AI 智能投資顧問")
    
    # 建立主分頁
    tab_analysis, tab_chat, tab_prediction_check = st.tabs([
        "📊 市場掃描與分析", 
        "💬 與 AI 分析師對話", 
        "✅ 每日預測驗證"
    ])

    with tab_analysis:
        render_stock_analysis(ai_advisor, symbol, hist_data, current_price, account)
    
    with tab_chat:
        render_ai_chat(symbol, current_price, ai_advisor)
    
    with tab_prediction_check:
        render_prediction_verification(ai_advisor, df_stocks)

def render_prediction_verification(ai_advisor, df_stocks):
    """預測驗證子功能 (暫時保留在彙整層，直到 sub_verification.py 完整分離)"""
    st.subheader("🕵️‍♀️ 每日 AI 預測驗證中心")
    st.success("✅ 系統自動化排程已啟用：將於每日 08:30 自動預測、14:30 自動驗證。")
    
    if os.path.exists(ai_advisor.prediction_file):
        try:
            with open(ai_advisor.prediction_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            if records:
                df_pred = pd.DataFrame(records)
                st.dataframe(df_pred.tail(100), use_container_width=True, hide_index=True)
            else:
                st.info("尚無紀錄。")
        except Exception as e:
            st.error(f"讀取紀錄失敗: {e}")
