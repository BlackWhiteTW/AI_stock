import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_kline_chart(symbol, hist_data, current_price):
    """K 線圖與趨勢線繪製子模組"""
    st.subheader(f"{symbol} 走勢圖")

    # --- 時間區間選擇 ---
    col_range, col_dummy = st.columns([2, 1])
    with col_range:
        range_options = ["近1週", "近1月", "近1季", "近半年", "近1年", "全部"]
        selected_range = st.select_slider("顯示區間", options=range_options, value="近1季") 

    df = hist_data.copy()
    
    # 基礎 MA 計算
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 建立 Figure
    fig = go.Figure()

    # 1. K線
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a', 
        name="K線"
    ))
    
    # 2. 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#ffa726', width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#42a5f5', width=1.5), name="MA60"))
    
    fig.update_layout(
        height=600,
        dragmode="pan",
        xaxis_title="日期",
        yaxis_title="價格",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.1), type="date")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 提示：使用滑動條調整顯示範圍。")
