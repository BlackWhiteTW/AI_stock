"""
🔍 搜尋優化服務 (Search Optimizer) v1.0
參考 FinAgent 邏輯，整合技術、基本、消息面與 AI 診斷的全方位搜尋引擎
"""
import pandas as pd
import streamlit as st
from datetime import datetime
from services.market_data import MarketDataManager
from services.financial_data import FinancialDataManager
from services.market_sentiment import MarketSentimentAnalyzer
from services.news_analyzer import NewsAnalyzer
from services.llm_service import LLMService
from services.search_engine import SearchEngine # 🆕 聯網搜尋引擎

class SearchOptimizer:
    def __init__(self):
        self.market_data = MarketDataManager()
        self.financial_manager = FinancialDataManager()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.llm = LLMService()
        self.search_engine = SearchEngine() # 🆕 搜尋引擎初始化

    def get_smart_summary(self, ticker, ai_advisor=None):
        """
        獲取該股票的全方位智慧摘要 (完全體: 技術+基本+內建新聞+SerpAPI聯網新聞+Gemini)
        """
        # 1. 基礎數據載入 (省略部分代碼)
        hist = self.market_data.load_data(ticker)
        if hist.empty: return None
        current_price = hist['Close'].iloc[-1]
        
        # 2. 技術面與基本面
        trend = self.sentiment_analyzer.get_multi_timeframe_trend(ticker)
        revenue_data = self.financial_manager.get_revenue_data(ticker)
        
        # 3. [優化] 聯網搜尋即時新聞 (SerpAPI)
        # 取得股票名稱以便搜尋
        stock_name = ticker # Default
        try:
            from utils import get_tw_stock_list
            df_list = get_tw_stock_list()
            name_row = df_list[df_list['Ticker'] == ticker]
            if not name_row.empty:
                stock_name = name_row.iloc[0]['名稱']
        except: pass

        web_news = self.search_engine.get_latest_news(f"{stock_name} 股票 股價分析")
        
        # 4. [優化] 整合所有情報給 Gemini
        tech_context = f"趨勢: {trend['daily']}, MFI: {self.sentiment_analyzer.calculate_money_flow_index(hist).iloc[-1]:.1f}"
        
        ai_verdict = self.llm.generate_stock_analysis(
            stock_info=f"{stock_name} ({ticker}) 現價 {current_price}",
            technical_data=tech_context,
            fundamental_data=f"營收數據: {revenue_data.get('monthly_revenue', [{}])[0].get('單月營收年增率', '未知')}",
            news_summary=web_news # 🆕 這裡改用 SerpAPI 抓到的最新鮮消息
        )

        return {
            "ticker": ticker,
            "current_price": current_price,
            "industry": self.news_analyzer.detect_industry(ticker, ""),
            "web_news": web_news,
            "ai_verdict": ai_verdict,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    @st.cache_data(ttl=1800) # 30分鐘快取
    def get_cached_summary(_self, ticker, _ai_advisor=None):
        """帶快取的摘要獲取"""
        return _self.get_smart_summary(ticker, _ai_advisor)
