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
        # 1. 基礎數據載入
        hist = self.market_data.load_data(ticker)
        if hist.empty: return None
        current_price = hist['Close'].iloc[-1]
        
        # 2. 技術面與基本面分析
        trend = self.sentiment_analyzer.get_multi_timeframe_trend(ticker)
        mfi = self.sentiment_analyzer.calculate_money_flow_index(hist).iloc[-1]
        
        # 獲取營收與財務評分
        revenue_data = self.financial_manager.get_revenue_data(ticker)
        health_score = self.financial_manager.get_financial_health_score(ticker)
        
        # 計算營收成長率 (Fallback 邏輯)
        revenue_growth = 0
        try:
            if revenue_data and 'monthly_revenue' in revenue_data:
                revenue_growth = float(revenue_data['monthly_revenue'][0].get('單月營收年增率', 0))
        except: pass

        # 3. [優化] 聯網搜尋即時新聞 (SerpAPI) 與 產業新聞 (RSS)
        # 取得股票名稱以便搜尋
        stock_name = ticker # Default
        try:
            from utils import get_tw_stock_list
            df_list = get_tw_stock_list()
            name_row = df_list[df_list['Ticker'] == ticker]
            if not name_row.empty:
                stock_name = name_row.iloc[0]['名稱']
        except: pass

        # A. 個股聯網搜尋 (SerpAPI)
        web_news = self.search_engine.get_latest_news(f"{stock_name} 股票 股價分析")
        
        # B. 🆕 產業新聞與情緒分析 (RSS + 國際新聞)
        industry = self.news_analyzer.detect_industry(ticker, stock_name)
        industry_score = 0
        industry_summary = "無產業新聞"
        industry_news_text = ""
        
        if industry:
            score, summary, details = self.news_analyzer.analyze_industry_sentiment(industry, detailed=True)
            industry_score = score
            industry_summary = summary
            # 格式化產業新聞供 LLM 參考
            if details.get('matched_news'):
                industry_news_text = "\n".join([f"- {n['title']} (Score: {n['score']})" for n in details['matched_news'][:5]])

        # C. 綜合情緒計算 (個股新聞情緒 + 產業情緒)
        # 這裡我們結合既有的 analyze_sentiment (個股) 與新的 industry_score
        stock_sentiment_score, _ = self.news_analyzer.analyze_sentiment(ticker)
        # 綜合分數權重：個股 70%, 產業 30%
        combined_sentiment_score = (stock_sentiment_score * 0.7) + (industry_score * 0.3)

        # 4. [優化] 整合所有情報給 Gemini
        tech_context = f"趨勢: {trend['daily']}, MFI: {mfi:.1f}, 大盤情緒: {self.sentiment_analyzer.get_market_sentiment()['sentiment']}"
        fundamental_context = f"營收成長: {revenue_growth:.1f}%, 財務健康度: {health_score:.0f}/100"
        
        combined_news_context = f"【個股新聞】:\n{web_news}\n\n【產業趨勢 ({industry})】:\n{industry_summary}\n{industry_news_text}"
        
        ai_verdict = self.llm.generate_stock_analysis(
            stock_info=f"{stock_name} ({ticker}) 現價 {current_price}",
            technical_data=tech_context,
            fundamental_data=fundamental_context,
            news_summary=combined_news_context
        )

        # 返回 UI 預期的結構
        return {
            "ticker": ticker,
            "current_price": current_price,
            "industry": industry,
            "sentiment": {
                "score": combined_sentiment_score,
                "level": self.news_analyzer.get_sentiment_level(combined_sentiment_score),
                "industry_score": industry_score
            },
            "technical": {
                "trend": trend,
                "mfi": mfi
            },
            "fundamental": {
                "revenue_growth": revenue_growth,
                "health_score": health_score
            },
            "web_news": web_news,
            "industry_summary": industry_summary,
            "ai_verdict": ai_verdict,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    @st.cache_data(ttl=1800) # 30分鐘快取
    def get_cached_summary(_self, ticker, _ai_advisor=None):
        """帶快取的摘要獲取"""
        return _self.get_smart_summary(ticker, _ai_advisor)
