import streamlit as st
import pandas as pd
from pages_ui.components.shared import header_with_icon, status_card

def render_portfolio_overview(account, current_price_map):
    """資產庫存概況子模組"""
    header_with_icon("我的資產庫存", "💰", "實時監控持股損益與分布")
    
    if not account.portfolio:
        st.info("目前尚無持倉。")
        return

    # 計算總資產
    total_market_value = 0
    portfolio_data = []
    
    for ticker, info in account.portfolio.items():
        qty = info['qty']
        avg_cost = info['avg_price']
        curr_p = current_price_map.get(ticker, avg_cost)
        market_value = qty * curr_p
        profit = (curr_p - avg_cost) * qty
        profit_pct = (curr_p - avg_cost) / avg_cost * 100
        
        total_market_value += market_value
        portfolio_data.append({
            "股票": ticker,
            "數量": qty,
            "平均成本": avg_cost,
            "當前價格": curr_p,
            "市值": market_value,
            "損益": profit,
            "報酬率 (%)": round(profit_pct, 2)
        })
    
    # 顯示統計卡片
    c1, c2, c3 = st.columns(3)
    c1.metric("帳戶餘額", f"${account.balance:,.0f}")
    c2.metric("股票市值", f"${total_market_value:,.0f}")
    c3.metric("總資產", f"${(account.balance + total_market_value):,.0f}")
    
    # 顯示表格
    df_p = pd.DataFrame(portfolio_data)
    st.dataframe(df_p, use_container_width=True, hide_index=True)
