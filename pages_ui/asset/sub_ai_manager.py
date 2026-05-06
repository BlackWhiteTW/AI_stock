import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon

def render_ai_investment_manager(account, ai_advisor, current_price_map):
    """AI 投資經理子模組"""
    header_with_icon("AI 投資經理決策", "", "AI 自動分析您的投資組合並提供優化建議")
    
    if not account.portfolio:
        st.info(" 目前無持倉，AI 經理建議：可從小部位開始嘗試市場掃描到的優質標的。")
        return

    st.subheader(" AI 投資組合分析報告")
    with st.spinner("AI 正在評估您的持股結構..."):
        # 這裡會模擬調用 ai_advisor 進行組合風險與收益評估
        portfolio_summary = _get_portfolio_analysis(account, ai_advisor, current_price_map)
        
        st.markdown(portfolio_summary)
        
    st.divider()
    st.markdown("### ️ 組合調整建議")
    st.info("建議減少高波標的比例，並適度增加防禦性資產以對抗目前市場波動。")

def _get_portfolio_analysis(account, ai_advisor, current_price_map):
    """內部模擬組合分析"""
    # 這裡實務上會組合所有持股資訊傳給 LLM
    tickers = list(account.portfolio.keys())
    return f"目前您的組合包含 {len(tickers)} 檔股票，集中在 {tickers[0] if tickers else '未知'} 等標的。總體風險係數為 中等。"
