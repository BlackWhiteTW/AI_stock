"""
🔍 搜尋優化服務 (Search Optimizer) v1.0
參考 FinAgent 邏輯，整合技術、基本、消息面與 AI 診斷的全方位搜尋引擎
"""

import pandas as pd
import streamlit as st
import os
import json
from datetime import datetime, timedelta
from services.market_data import MarketDataManager
from services.financial_data import FinancialDataManager
from services.market_sentiment import MarketSentimentAnalyzer
from services.news_analyzer import NewsAnalyzer
from services.llm_service import LLMService
from services.search_engine import SearchEngine  # 🆕 聯網搜尋引擎


class SearchOptimizer:
    def __init__(self):
        self.market_data = MarketDataManager()
        self.financial_manager = FinancialDataManager()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.llm = LLMService()
        self.search_engine = SearchEngine()  # 🆕 搜尋引擎初始化

        # 🚀 優化：對接統一快取管理器
        from .cache_manager import get_cache_manager

        self.cache_manager = get_cache_manager()

    def get_smart_summary(self, ticker, ai_advisor=None, force_update=False):
        """
        獲取該股票的全方位智慧摘要 (完全體: 技術+基本+內建新聞+SerpAPI聯網新聞+Gemini)
        """
        print(f"🔍 [AI 快報] 開始分析股票: {ticker}")

        # 0. 🚀 使用統一快取獲取 (24小時)
        if not force_update:
            try:
                cached_result = self.cache_manager.get("analysis_report", ticker)
                if cached_result:
                    print(f"⚡ [AI 快報] 從快取讀取成功: {ticker}")
                    return cached_result
            except Exception as e:
                print(f"⚠️ 快取讀取異常: {e}")

        # 1. 基礎數據載入 (本地或快取)
        print(f"📊 [1/5] 正在載入基礎數據...")
        try:
            hist = self.market_data.load_data(ticker)
            if hist.empty:
                print(f"⚠️ 找不到 {ticker} 的歷史數據")
                # ⚡ 返回基本結構而不是 None
                return {
                    "ticker": ticker,
                    "current_price": 0,
                    "industry": "未知",
                    "web_news": "找不到該股票數據",
                    "industry_summary": "無數據",
                    "sentiment": {"score": 0, "level": "中性", "industry_score": 0},
                    "technical": {"trend": {"daily": "Unknown"}, "mfi": 50},
                    "fundamental": {"revenue_growth": 0, "health_score": 0},
                    "ai_verdict": f"⚠️ 找不到股票 {ticker} 的數據，請確認股票代碼是否正確",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            current_price = hist["Close"].iloc[-1]
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            # ⚡ 返回基本結構而不是 None
            return {
                "ticker": ticker,
                "current_price": 0,
                "industry": "未知",
                "web_news": "數據載入失敗",
                "industry_summary": "無數據",
                "sentiment": {"score": 0, "level": "中性", "industry_score": 0},
                "technical": {"trend": {"daily": "Unknown"}, "mfi": 50},
                "fundamental": {"revenue_growth": 0, "health_score": 0},
                "ai_verdict": f"❌ 數據載入失敗: {str(e)[:100]}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }

        # 2. 技術面與基本面分析 (通常極快)
        print(f"📈 [2/5] 正在進行技術面分析...")
        trend = self.sentiment_analyzer.get_multi_timeframe_trend(ticker)
        mfi = self.sentiment_analyzer.calculate_money_flow_index(hist).iloc[-1]
        revenue_data = self.financial_manager.get_revenue_data(ticker)
        health_score = self.financial_manager.get_financial_health_score(ticker)

        revenue_growth = 0
        try:
            if revenue_data and "monthly_revenue" in revenue_data:
                revenue_growth = float(
                    revenue_data["monthly_revenue"][0].get("單月營收年增率", 0)
                )
        except:
            pass

        # 3. 聯網搜尋即時新聞與產業分析 (設定 8 秒硬性超時)
        print(f"🌐 [3/5] 正在搜尋聯網新聞 (限時 8s)...")
        web_news = "暫無即時新聞"
        industry = None
        industry_score = 0
        industry_summary = "無產業新聞"

        import threading

        def fetch_worker():
            nonlocal web_news, industry, industry_score, industry_summary
            try:
                stock_name = ticker
                try:
                    from utils import get_tw_stock_list

                    df_list = get_tw_stock_list()
                    name_row = df_list[df_list["Ticker"] == ticker]
                    if not name_row.empty:
                        stock_name = name_row.iloc[0]["名稱"]
                except:
                    pass

                search_query = f"{stock_name} {ticker} 股票 最新消息"
                news_result = self.search_engine.get_latest_news(search_query)
                # ⚡ 防止 None 覆盖默认值
                if news_result:
                    web_news = news_result

                industry = self.news_analyzer.detect_industry(ticker, stock_name)
                if industry:
                    score, summary, _ = self.news_analyzer.analyze_industry_sentiment(
                        industry, detailed=True
                    )
                    industry_score = score
                    industry_summary = summary
            except:
                pass

        t = threading.Thread(target=fetch_worker)
        t.start()
        t.join(timeout=8.0)
        if t.is_alive():
            print("⏳ [AI 快報] 聯網搜尋超時，跳過以維持 UI 反應。")

        # ⚡ 確保 web_news 不為 None
        if web_news is None:
            web_news = "暫無即時新聞"

        # 綜合情緒
        stock_sentiment_score, _ = self.news_analyzer.analyze_sentiment(ticker)
        combined_sentiment_score = (stock_sentiment_score * 0.7) + (
            industry_score * 0.3
        )

        # 4. 呼叫 Gemini (設定 30 秒硬性超時)
        print(f"🤖 [4/5] 正在呼叫 Gemini 生成報告 (限時 30s)...")
        tech_context = f"趨勢: {trend['daily']}, MFI: {mfi:.1f}"
        fundamental_context = f"營收成長: {revenue_growth:.1f}%"
        # ⚡ 安全切片：确保 web_news 是字符串
        news_preview = str(web_news)[:200] if web_news else "暫無新聞"
        industry_preview = str(industry_summary) if industry_summary else "無產業資訊"
        combined_news_context = f"新聞: {news_preview}...\n產業: {industry_preview}"

        ai_verdict = "⚠️ AI 生成超時，請稍後再試。"

        def ai_worker():
            nonlocal ai_verdict
            try:
                ai_verdict = self.llm.generate_stock_analysis(
                    stock_info=f"{ticker} 現價 {current_price}",
                    technical_data=tech_context,
                    fundamental_data=fundamental_context,
                    news_summary=combined_news_context,
                )
            except:
                pass

        at = threading.Thread(target=ai_worker)
        at.start()
        at.join(timeout=30.0)
        if at.is_alive():
            print("⏳ [AI 快報] Gemini 調用超時。")

        # 5. 組合結果並存入快取
        print(f"✅ [5/5] 分析完成")
        result = {
            "ticker": ticker,
            "current_price": current_price,
            "industry": industry or "未知",
            "web_news": web_news or "暫無即時新聞",  # ⚡ 添加 web_news 字段
            "industry_summary": industry_summary or "無產業資訊",  # ⚡ 添加產業摘要
            "sentiment": {
                "score": combined_sentiment_score,
                "level": self.news_analyzer.get_sentiment_level(
                    combined_sentiment_score
                ),
                "industry_score": industry_score,  # ⚡ 添加產業情緒分數
            },
            "technical": {"trend": trend, "mfi": mfi},
            "fundamental": {
                "revenue_growth": revenue_growth,
                "health_score": health_score,
            },
            "ai_verdict": ai_verdict or "⚠️ AI 分析暫時無法使用",  # ⚡ 確保不為 None
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

        try:
            self.cache_manager.set("analysis_report", ticker, result, ttl=24 * 3600)
        except:
            pass

        return result

    # @st.cache_data 移除，因為它在快取失效時會阻塞整個 Streamlit UI 渲染
    def get_cached_summary(self, ticker, ai_advisor=None):
        """帶快取的摘要獲取 (非阻塞調用)"""
        try:
            return self.get_smart_summary(ticker, ai_advisor, force_update=False)
        except Exception as e:
            print(f"⚠️ get_cached_summary 失敗: {e}")
            return {
                "ticker": ticker,
                "current_price": 0,
                "industry": "未知",
                "web_news": "暫時無法獲取",
                "industry_summary": "暫時無法獲取",
                "sentiment": {"score": 0, "level": "中性", "industry_score": 0},
                "technical": {"trend": {"daily": "Unknown"}, "mfi": 50},
                "fundamental": {"revenue_growth": 0, "health_score": 0},
                "ai_verdict": f"⚠️ 分析服務暫時無法使用: {str(e)[:100]}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
