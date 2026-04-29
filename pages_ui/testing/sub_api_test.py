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
        if st.button("測試 SerpAPI (Google Search) 連線", key="test_serp"):
            _test_serpapi()

    st.divider()
    st.subheader("📰 周期性新聞收集器")
    st.caption("檢查與手動更新 `data/weekly_news` 資料夾中的新聞數據")
    
    col_news1, col_news2 = st.columns(2)
    
    from services.weekly_news_collector import WeeklyNewsCollector
    collector = WeeklyNewsCollector()
    
    with col_news1:
        if st.button("🚀 執行手動新聞收集", key="run_news_collect"):
            with st.spinner("正在搜尋並分析新聞..."):
                try:
                    stats = collector.collect_daily_news()
                    st.success(f"✅ 收集完成！本次新增 {stats['news_collected_today']} 則新聞。")
                    st.json(stats)
                except Exception as e:
                    st.error(f"❌ 收集失敗: {e}")
    
    with col_news2:
        if st.button("📊 查看本周收集摘要", key="view_news_summary"):
            summary = collector.get_recent_news_summary(days=7)
            st.write(f"📅 **本周累積新聞數**: {summary['total_news']} 則")
            st.write(f"🧠 **平均市場情緒**: {summary['average_sentiment']:.2f}")
            if summary['by_category']:
                st.write("**類別統計**:")
                st.json(summary['by_category'])
            else:
                st.warning("⚠️ 本周尚無新聞數據，請先執行收集。")

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

def _test_serpapi():
    from services.search_engine import SearchEngine
    engine = SearchEngine()
    start_time = time.time()
    with st.spinner("正在呼叫 SerpAPI 搜尋台積電最新消息..."):
        try:
            res = engine.get_latest_news("台積電 股價分析")
            elapsed = time.time() - start_time
            if res and "❌" not in res and "⚠️" not in res:
                st.success(f"✅ SerpAPI 正常！(耗時: {elapsed:.2f}s)")
                st.write("**搜尋結果摘要**:")
                st.write(res)
            else:
                st.error(f"❌ SerpAPI 連線異常: {res}")
        except Exception as e:
            st.error(f"❌ SerpAPI 呼叫崩潰: {e}")
