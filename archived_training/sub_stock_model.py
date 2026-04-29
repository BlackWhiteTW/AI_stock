import streamlit as st
import time
from datetime import datetime
from pages_ui.components.shared import header_with_icon, status_card
from utils import get_stock_data


def render_stock_model_training(ai_advisor):
    """個股模型訓練子模組"""
    header_with_icon(
        "個股深度學習模型訓練", "📈", "針對特定股票訓練專屬的 GRU/LSTM 模型"
    )

    ticker_input = st.text_input("輸入股票代碼", placeholder="例如: 2330, 2454, 0050")

    if ticker_input:
        ticker = ticker_input.replace(".TW", "").replace(".TWO", "")
        if st.button("🧠 啟動單股訓練", type="primary"):
            _run_stock_training(ai_advisor, ticker)


def _run_stock_training(ai_advisor, ticker):
    """執行單股訓練"""
    with st.spinner(f"正在抓取 {ticker} 數據並建立模型..."):
        _, hist, _, _ = get_stock_data(ticker)
        if not hist.empty:
            msg, loss = ai_advisor.train_deep_learning_model(ticker, hist)
            st.success(f"✅ {ticker} 訓練完成！Loss: {loss:.6f}")
            st.balloons()
        else:
            st.error("❌ 無法獲取股票數據")
