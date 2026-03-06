import streamlit as st
import os
import datetime
import pandas as pd
from services.market_scanner_helper import MarketScannerUI, create_scan_mode_selector

def render_stock_analysis(ai_advisor, symbol, hist_data, current_price, account):
    """個股 AI 診斷與預測子模組"""
    col_ai_1, col_ai_2 = st.columns([1, 1])
    
    with col_ai_1:
        st.subheader(f"🎯 目標個股分析: {symbol}")
        if not hist_data.empty:
            current_params = ai_advisor._get_params(symbol)
            st.caption(f"當前模式: {current_params.get('name', '標準均衡型 (預設)')}")

            if st.button("🧠 開始分析此股票", key="btn_analyze"):
                _execute_analysis(ai_advisor, symbol, hist_data, current_price, account)

    with col_ai_2:
        st.subheader("📡 市場掃描雷達")
        st.info("AI 將掃描您的 **自選清單 (Watchlist)**，找出出現 **買入訊號** 的潛力股。")
        if st.session_state.user_watchlist:
            include_news, use_cache = create_scan_mode_selector()
            if st.button("🚀 啟動掃描 (自選股)"):
                _execute_watchlist_scan(ai_advisor, include_news, use_cache)
        else:
            st.write("👀 您的自選清單是空的。")

def _execute_analysis(ai_advisor, symbol, hist_data, current_price, account):
    """執行分析、訓練與預測邏輯"""
    with st.spinner("AI 正在計算數據..."):
        result = ai_advisor.analyze_stock(symbol, hist_data)
        st.divider()
        st.metric(label="AI 建議方向", value=result['action'], delta=f"信心分數: {result['score']}")
        
        with st.expander("📊 查看分析理由", expanded=True):
            for r in result['reasons']:
                st.write(f"- {r}")
        
        # 預測邏輯
        pred_price = ai_advisor.predict_next_price(symbol, hist_data)
        if pred_price:
            diff = pred_price - current_price
            st.metric("🔮 AI 預估次日價格", f"{pred_price:.2f}", f"{diff:+.2f} ({(diff/current_price)*100:+.2f}%)")

def _execute_watchlist_scan(ai_advisor, include_news, use_cache):
    """執行自選股掃描"""
    scanner_ui = MarketScannerUI(ai_advisor)
    results, scan_time = scanner_ui.scan_with_progress(
        tickers=list(st.session_state.user_watchlist.keys()),
        include_news=include_news,
        use_cache=use_cache
    )
    scanner_ui.display_results(results, scan_time)
