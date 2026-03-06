import streamlit as st
from .analysis.sub_kline import render_kline_chart

def render_tab_analysis(symbol, hist_data, current_price):
    """
    K 線圖與技術分析 (彙整入口)
    """
    if not hist_data.empty:
        render_kline_chart(symbol, hist_data, current_price)
    else:
        st.warning(f"找不到代碼 {symbol} 的數據。")
