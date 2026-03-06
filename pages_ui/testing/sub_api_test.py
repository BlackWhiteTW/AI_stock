import streamlit as st
import time
import requests
import os
from services.llm_service import LLMService
from pages_ui.components.shared import header_with_icon

def render_api_testing():
    """API 連線診斷子模組"""
    header_with_icon("系統服務診斷", "🔧", "檢查 AI 模型與外部數據源的連線狀態")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI 服務測試")
        if st.button("測試 Gemini API 連線", key="test_gemini"):
            _test_gemini()
            
    with col2:
        st.subheader("🌐 數據源測試")
        if st.button("測試 Yahoo Finance 連線", key="test_yf"):
            _test_yfinance()

def _test_gemini():
    llm = LLMService()
    start_time = time.time()
    with st.spinner("正在發送測試請求至 Google..."):
        try:
            res = llm.call_gemini("你好，請回傳『連線成功』四個字。")
            elapsed = time.time() - start_time
            if "連線成功" in res:
                st.success(f"✅ Gemini API 正常！(耗時: {elapsed:.2f}s)")
                st.info(f"當前使用模型: {llm.model_name}")
            else:
                st.warning(f"⚠️ 連線回傳異常: {res}")
        except Exception as e:
            st.error(f"❌ Gemini 連線失敗: {e}")

def _test_yfinance():
    with st.spinner("正在從 Yahoo 下載 2330.TW 測試數據..."):
        try:
            import yfinance as yf
            data = yf.download("2330.TW", period="5d", progress=False)
            if not data.empty:
                st.success(f"✅ Yahoo Finance 正常！已獲取 {len(data)} 筆資料。")
                st.line_chart(data['Close'])
            else:
                st.error("❌ Yahoo Finance 回傳數據為空。")
        except Exception as e:
            st.error(f"❌ 數據下載失敗: {e}")
