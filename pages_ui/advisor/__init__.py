import streamlit as st
from .sub_analysis import render_stock_analysis
from .sub_chat import render_ai_chat
from .sub_verification import render_prediction_analysis

def render_tab_advisor(ai_advisor, symbol, hist_data, current_price, account, df_stocks):
    st.header("🤖 AI 智能投資顧問")
    tab_analysis, tab_chat, tab_prediction_check = st.tabs(["📊 市場掃描與分析", "💬 與 AI 分析師對話", "✅ 每日預測驗證"])

    with tab_analysis:
        render_stock_analysis(ai_advisor, symbol, hist_data, current_price, account)
    
    with tab_chat:
        render_ai_chat(symbol, current_price, ai_advisor)
    
    with tab_prediction_check:
        render_prediction_analysis(ai_advisor, df_stocks)
