"""
性能優化工具模組
提供數據批量處理、並行計算、智能快取等功能
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Callable, Any
import time
import hashlib
import pickle
from datetime import datetime, timedelta
import threading

class BatchDataLoader:
    """批量數據加載器 - 優化多股票數據獲取"""
    
    def __init__(self, data_manager, max_workers=8):
        """
        初始化批量加載器
        :param data_manager: MarketDataManager 實例
        :param max_workers: 最大線程數
        """
        self.data_manager = data_manager
        self.max_workers = max_workers
        self._cache_lock = threading.Lock()
        
    def load_multiple_stocks(self, tickers: List[str], only_cached=False, progress_callback=None):
        """
        並行加載多個股票數據
        :param tickers: 股票代碼列表
        :param only_cached: 是否只讀取快取
        :param progress_callback: 進度回調函數 callback(current, total, ticker)
        :return: Dict[ticker, DataFrame]
        """
        results = {}
        total = len(tickers)
        completed = 0
        
        def load_single(ticker):
            try:
                df = self.data_manager.load_data(ticker, only_cached=only_cached)
                return ticker, df
            except Exception as e:
                return ticker, pd.DataFrame()
        
        # 使用線程池並行加載
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(load_single, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker, df = future.result()
                results[ticker] = df
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, ticker)
        
        return results


class TechnicalIndicatorCache:
    """⚡ 優化版技術指標快取系統 - 對接統一快取管理器"""
    
    def __init__(self, ttl_hours=24):
        from .cache_manager import get_cache_manager
        self.cache_manager = get_cache_manager()
        self.ttl = ttl_hours * 3600
        
    def _get_identifier(self, ticker, params, data_hash):
        """生成唯一識別符，包含參數與數據指紋"""
        param_str = str(sorted(params.items()))
        combined_hash = hashlib.md5(f"{param_str}_{data_hash}".encode()).hexdigest()
        return f"{ticker}_{combined_hash}"
    
    def get(self, ticker, params, data_hash):
        """獲取快取的指標數據"""
        identifier = self._get_identifier(ticker, params, data_hash)
        return self.cache_manager.get("technical_indicators", identifier)
    
    def set(self, ticker, params, df, data_hash):
        """保存指標數據到快取"""
        identifier = self._get_identifier(ticker, params, data_hash)
        # 利用 CacheManager 的自動判定，DataFrame 會存為 Parquet
        self.cache_manager.set("technical_indicators", identifier, df, ttl=self.ttl)
    
    def clear_expired(self):
        """清理過期快取 (調用統一清理)"""
        self.cache_manager.cleanup_expired()


class ParallelStockScanner:
    """並行股票掃描器 - 用於購物清單功能"""
    
    def __init__(self, ai_advisor, max_workers=6):
        """
        初始化掃描器
        :param ai_advisor: AIAdvisor 實例
        :param max_workers: 最大線程數
        """
        self.ai_advisor = ai_advisor
        self.max_workers = max_workers
        
    def scan_stocks(self, tickers: List[str], hist_data_dict: Dict[str, pd.DataFrame], 
                    include_news=True, progress_callback=None):
        """
        並行掃描多個股票
        :param tickers: 股票代碼列表
        :param hist_data_dict: 預先加載的歷史數據字典
        :param include_news: 是否包含新聞分析
        :param progress_callback: 進度回調 callback(current, total, ticker, result)
        :return: List[Dict] 掃描結果列表
        """
        results = []
        total = len(tickers)
        completed = 0
        
        def scan_single(ticker):
            """掃描單個股票"""
            try:
                hist = hist_data_dict.get(ticker, pd.DataFrame())
                if hist.empty or len(hist) < 60:  # 提高數據要求到 60 天
                    return None
                
                # 【優化】預篩選：快速排除明顯不符合的股票
                current_price = hist['Close'].iloc[-1]
                ma20 = hist['Close'].rolling(20).mean().iloc[-1]
                ma60 = hist['Close'].rolling(60).mean().iloc[-1]
                avg_volume = hist['Volume'].tail(20).mean()
                
                # 基本條件：
                # 1. 價格在 MA20 之上 (短期趨勢向上)
                # 2. MA20 在 MA60 之上 (多頭排列)
                # 3. 成交量至少 300 張以上 (有足夠流動性)
                if current_price < ma20 or ma20 < ma60 or avg_volume < 300000:
                    return None
                
                # 技術分析
                analysis = self.ai_advisor.analyze_stock(ticker, hist)
                
                # 新聞分析(可選)
                news_score = 0
                news_summary = ""
                if include_news:
                    try:
                        news_result = self.ai_advisor.news_analyzer.analyze_stock_news(ticker, limit=5)
                        news_score = news_result.get('overall_score', 0)
                        news_summary = news_result.get('summary', '')
                    except:
                        pass
                
                # 組合結果
                result = {
                    '代碼': ticker,
                    'AI評分': analysis.get('score', 0),
                    '建議': analysis.get('action', '觀望'),
                    '新聞分數': news_score,
                    '新聞摘要': news_summary[:50] + '...' if len(news_summary) > 50 else news_summary,
                    '技術理由': ', '.join(analysis.get('reasons', [])[:2])
                }
                
                return result
                
            except Exception as e:
                return None
        
        # 使用線程池並行掃描
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(scan_single, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                result = future.result()
                
                if result:
                    results.append(result)
                
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, ticker, result)
        
        return results

    def scan_stocks_parallel(self, stock_data_list: List[Tuple[str, pd.DataFrame]], 
                            include_news=True, use_cache=True):
        """
        並行分析股票數據 (為 MarketScannerHelper 提供)
        :param stock_data_list: [(ticker, df), ...] 元組列表
        :param include_news: 是否包含新聞
        :param use_cache: 是否使用快取 (此處主要依賴 Advisor 內部的快取)
        :return: Dict[str, Dict] {ticker: analysis_result}
        """
        results = {}
        
        def analyze_single(item):
            ticker, df = item
            try:
                # 委託給 AIAdvisor 進行分析
                analysis = self.ai_advisor.analyze_stock(ticker, df, include_news=include_news)
                return ticker, analysis
            except Exception as e:
                print(f"並行分析 {ticker} 失敗: {e}")
                return ticker, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(analyze_single, item) for item in stock_data_list]
            for future in as_completed(futures):
                ticker, analysis = future.result()
                if analysis:
                    results[ticker] = analysis
                    
        return results


class StreamlitCacheHelper:
    """Streamlit 快取輔助工具"""
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """計算 DataFrame 的哈希值"""
        try:
            # 使用最後一行數據+長度作為簡單哈希
            if df.empty:
                return "empty"
            last_row = df.iloc[-1].to_dict()
            hash_str = f"{len(df)}_{last_row}"
            return hashlib.md5(str(hash_str).encode()).hexdigest()[:16]
        except:
            return str(time.time())
    
    @staticmethod
    def should_recalculate(last_update_time, ttl_seconds=300):
        """
        判斷是否需要重新計算
        :param last_update_time: 上次更新時間戳
        :param ttl_seconds: 有效期(秒)
        :return: bool
        """
        if last_update_time is None:
            return True
        return (time.time() - last_update_time) > ttl_seconds


class NewsCache:
    """新聞快取優化器"""
    
    def __init__(self, cache_dir='data/news_cache', ttl_minutes=30):
        self.cache_dir = cache_dir
        self.ttl_minutes = ttl_minutes
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, ticker):
        """獲取快取的新聞"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            import json
            mtime = os.path.getmtime(cache_file)
            age_minutes = (time.time() - mtime) / 60
            
            if age_minutes > self.ttl_minutes:
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def set(self, ticker, news_data):
        """保存新聞到快取"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.json")
        
        try:
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=2)
        except:
            pass


# 全局實例(單例模式)
_batch_loader = None
_indicator_cache = None
_stock_scanner = None

def get_batch_loader(data_manager, max_workers=8):
    """獲取批量加載器單例"""
    global _batch_loader
    if _batch_loader is None:
        _batch_loader = BatchDataLoader(data_manager, max_workers)
    return _batch_loader

def get_indicator_cache():
    """獲取指標快取單例"""
    global _indicator_cache
    if _indicator_cache is None:
        _indicator_cache = TechnicalIndicatorCache()
    return _indicator_cache

def get_stock_scanner(ai_advisor, max_workers=6):
    """獲取股票掃描器單例"""
    global _stock_scanner
    if _stock_scanner is None:
        _stock_scanner = ParallelStockScanner(ai_advisor, max_workers)
    return _stock_scanner
