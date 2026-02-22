import pandas as pd
from services.news_analyzer import NewsAnalyzer
from services.ai_service import AIService
from services.stock_strategy import StockStrategy
from services import tech_analysis

class AIAdvisor:
    def __init__(self):
        # 初始化各個子模組
        self.news_analyzer = NewsAnalyzer()
        self.service = AIService()
        # 將策略物件與服務共享 (若需要)
        # 這裡我們確保 Service 用的是同一個 strategy 實例，以便共享 learned_params
        self.strategy = self.service.strategy 

    # --- 屬性代理 (Property Proxy) ---
    # 讓外部 UI 能夠像以前一樣訪問這兩個變數
    @property
    def models(self):
        return self.service.models
        
    @property
    def learned_params(self):
        return self.strategy.learned_params

    @property
    def prediction_file(self):
        return self.service.prediction_file

    @property
    def model_dir(self):
        return self.service.model_dir

    # --- 功能代理 (Method Proxy) ---

    def train_deep_learning_model(self, ticker, df, progress_callback=None):
        """訓練深度學習模型 (委派給 AIService)"""
        return self.service.train_deep_learning_model(ticker, df, progress_callback)

    def predict_next_price(self, ticker, df):
        """預測股價 (委派給 AIService)"""
        return self.service.predict_next_price(ticker, df)

    def get_backtest_data(self, ticker, df, days=60):
        """取得回測驗證數據 (委派給 AIService)"""
        return self.service.get_backtest_data(ticker, df, days)

    def run_morning_predictions(self):
        """執行每日晨間預測 (委派給 AIService)"""
        return self.service.run_morning_predictions()

    def validate_afternoon_results(self):
        """執行每日收盤驗證 (委派給 AIService)"""
        return self.service.validate_afternoon_results()

    def optimize_strategy_basic(self, ticker, df):
        """策略優化 (委派給 StockStrategy)"""
        return self.strategy.optimize_strategy_basic(ticker, df)

    def _get_params(self, ticker):
        """取得策略參數 (相容性包裝)"""
        return self.strategy.get_params(ticker)
        
    def calculate_technical_indicators(self, df, params=None):
        """計算技術指標 (相容性包裝)"""
        return tech_analysis.calculate_technical_indicators(df, params)

    def analyze_stock(self, ticker, df, include_news=True):
        """
        綜合分析：結合 AI 預測值與技術指標策略
        """
        # 1. 先取得 AI 預測價格
        ai_pred_price = self.service.predict_next_price(ticker, df)
        
        # 2. 進行策略分析 (將 NewsAnalyzer 傳入以便內部調用)
        analysis_result = self.strategy.analyze_stock(
            ticker, 
            df, 
            ai_pred_price=ai_pred_price, 
            news_analyzer=self.news_analyzer, 
            include_news=include_news
        )
        
        return analysis_result
