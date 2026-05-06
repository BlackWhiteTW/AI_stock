import streamlit as st
from .asset.sub_portfolio import render_portfolio_overview
from .asset.sub_ai_manager import render_ai_investment_manager

def render_tab_asset(account, symbol, current_price, ai_advisor):
    """
    資產概況 (彙整入口)
    """
    st.header(" 資產概況與 AI 經理")
    
    asset_tab1, asset_tab2 = st.tabs([" 我的庫存", " AI 投資經理"])
    
    # 建立目前價格映射 (實務上會從 API 抓取)
    current_price_map = {symbol: current_price}
    
    with asset_tab1:
        render_portfolio_overview(account, current_price_map)
        
    with asset_tab2:
        render_ai_investment_manager(account, ai_advisor, current_price_map)
