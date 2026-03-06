import streamlit as st
import pandas as pd
import traceback
from services.market_scanner_helper import MarketScannerUI, create_scan_mode_selector
from pages_ui.components.shared import header_with_icon

def render_market_scanner(df_stocks, ai_advisor):
    """市場掃描與採購中心子模組"""
    header_with_icon("AI 市場掃描與採購中心", "🔍", "全市場掃描，找出符合 AI 策略的潛力標的")
    
    st.info("📡 市場掃描雷達：AI 將掃描市場，找出出現 **買入訊號** 的潛力股。")
    
    # 使用共用的掃描模式選擇器
    include_news, use_cache = create_scan_mode_selector()
    
    if st.button("🚀 啟動全市場掃描", type="primary", use_container_width=True):
        tickers_list = df_stocks['Ticker'].tolist()
        
        # 使用優化的掃描輔助類
        scanner_ui = MarketScannerUI(ai_advisor)
        
        try:
            results, scan_time = scanner_ui.scan_with_progress(
                tickers=tickers_list,
                include_news=include_news,
                use_cache=use_cache,
                min_score=5,  # 全市場掃描只顯示強勢股
                use_parallel=True,
                max_workers=10
            )
            
            # 顯示結果
            scanner_ui.display_results(results, scan_time)
            
        except Exception as e:
            st.error(f"掃描失敗: {e}")
            with st.expander("查看錯誤詳情"):
                st.code(traceback.format_exc())
