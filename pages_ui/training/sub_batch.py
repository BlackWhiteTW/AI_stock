import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils import get_stock_data, get_tw_stock_list
from services.stock_strategy import StockStrategy
from pages_ui.components.shared import header_with_icon

def render_batch_training(ai_advisor):
    """批量股票模型訓練子模組"""
    header_with_icon("批量股票模型訓練", "📦", "自動執行多支股票的 AI 預測模型訓練")
    
    # 建立三種批量模式的列
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📦 批量訓練熱門股")
        popular_stocks = ["0050.TW", "0056.TW", "2330.TW", "2317.TW", "2454.TW"]
        st.caption(f"訓練標的: {', '.join([s.split('.')[0] for s in popular_stocks])}")
        if st.button("🚀 開始批量訓練熱門股", key="batch_popular"):
            _run_simple_batch(ai_advisor, popular_stocks)

    with col2:
        st.markdown("### 🔥 智能篩選訓練")
        st.caption("AI 自動篩選技術指標評分高 (趨勢向上) 的股票進行訓練")
        if st.button("🧠 啟動智能篩選訓練", key="batch_smart"):
            _run_smart_batch(ai_advisor)

    with col3:
        st.markdown("### 🏆 全部優良股訓練")
        st.caption("針對 ai_shopping_list_basic.csv 中的標的進行訓練")
        if st.button("🏁 開始訓練全部優良股", key="batch_shopping"):
            _run_shopping_batch(ai_advisor)

def _run_simple_batch(ai_advisor, tickers):
    progress_bar = st.progress(0)
    for idx, t in enumerate(tickers):
        st.write(f"正在訓練: {t}...")
        _, hist, _, _ = get_stock_data(t)
        if not hist.empty:
            ai_advisor.train_deep_learning_model(t, hist)
        progress_bar.progress((idx + 1) / len(tickers))
    st.success("✅ 批量熱門股訓練完成！")

def _run_smart_batch(ai_advisor):
    df_stocks = get_tw_stock_list()
    tickers = df_stocks['Ticker'].tolist()
    analyzer = StockStrategy()
    
    success_count = 0
    progress_bar = st.progress(0)
    
    with st.status("正在進行全市場掃描與訓練...", expanded=True) as status:
        for idx, t in enumerate(tickers):
            _, hist, _, _ = get_stock_data(t, force_offline=True)
            if not hist.empty and len(hist) >= 60:
                result = analyzer.analyze_stock(t, hist)
                if result.get('score', 0) >= 5:
                    status.write(f"✨ {t} 評分達標 ({result['score']})，啟動訓練...")
                    ai_advisor.train_deep_learning_model(t, hist)
                    success_count += 1
            progress_bar.progress((idx + 1) / len(tickers))
    st.success(f"🎉 智能訓練完成！共訓練了 {success_count} 支強勢股。")

def _run_shopping_batch(ai_advisor):
    try:
        df = pd.read_csv("data/ai_shopping_list_basic.csv")
        tickers = [str(t).replace('.TW', '').replace('.TWO', '') for t in df['代碼'].tolist()]
        _run_simple_batch(ai_advisor, tickers)
    except Exception as e:
        st.error(f"❌ 無法讀取清單: {e}")
