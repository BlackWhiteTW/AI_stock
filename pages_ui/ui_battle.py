import streamlit as st
from .battle.sub_scanner import render_market_scanner
from .battle.sub_trader import render_trader_dashboard

def render_tab_battle(trader_bot_v1, trader_bot_v2, symbol, hist_data, df_stocks, ai_advisor):
    """
    AI 自動化股票交易 (彙整入口)
    """
    st.header("⚔️ AI 自動化交易中心")
    
    battle_tab1, battle_tab2 = st.tabs(["🔍 市場掃描與採購", "🤖 機器人控制面板"])
    
    with battle_tab1:
        render_market_scanner(df_stocks, ai_advisor)
        
    with battle_tab2:
        render_trader_dashboard(trader_bot_v1, trader_bot_v2)
