"""
AI 預測加速優化模組
=====================

針對 run_morning_predictions() 進行性能優化:
1. 🚀 多進程並行處理 (繞過 GIL，真正並行)
2. 💾 智能快取機制 (避免重複計算)
3. 🎯 優先級處理 (重要股票優先)
4. ⚡ 自動偵測 CPU 核心數
"""

import pandas as pd
import numpy as np
import torch
import concurrent.futures
import multiprocessing
import time
import os
import json
import datetime
from collections import defaultdict
from utils import get_stock_data

class FastPredictionEngine:
    """
    快速預測引擎 v2.0 - 多進程版本
    
    優化策略:
    - 多進程並行: 繞過 Python GIL 限制，真正並行計算
    - 批量載入: 一次載入多個模型
    - 快取結果: 避免重複計算
    - 自動調整: 根據 CPU 核心數自動設定並發數
    """
    
    def __init__(self, ai_service):
        self.ai_service = ai_service
        # 自動偵測 CPU 核心數，使用 80% 的核心
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max(4, int(cpu_count * 0.8))
        print(f"🖥️ 偵測到 {cpu_count} 核心，使用 {self.max_workers} 個並行進程")
        self.cache_ttl = 3600
        self.cache = {}
        
    def _predict_single_stock(self, ticker):
        """單一股票預測 (線程安全)"""
        try:
            # 檢查快取
            if ticker in self.cache:
                cache_time = self.cache[ticker]['time']
                if time.time() - cache_time < self.cache_ttl:
                    return self.cache[ticker]['result']
            
            # 獲取數據
            symbol, hist_data, current_price, _ = get_stock_data(ticker)
            
            if hist_data.empty:
                return None
            
            # 執行預測
            pred_price = self.ai_service.predict_next_price(ticker, hist_data)
            
            if not pred_price:
                return None
            
            # 計算基準價格
            today = datetime.datetime.now().date()
            hist_data.index = pd.to_datetime(hist_data.index)
            before_today = hist_data[hist_data.index.date < today]
            
            if not before_today.empty:
                base_price = float(before_today['Close'].iloc[-1])
            else:
                base_price = float(hist_data['Close'].iloc[-1])
            
            expected_change = ((pred_price - base_price) / base_price) * 100
            
            result = {
                "ticker": ticker,
                "base_price": base_price,
                "predicted_price": float(pred_price),
                "expected_change_pct": round(expected_change, 2)
            }
            
            # 存入快取
            self.cache[ticker] = {
                'time': time.time(),
                'result': result
            }
            
            return result
        
        except Exception as e:
            print(f"預測 {ticker} 失敗: {e}")
            return None
    
    def batch_predict(self, ticker_list, priority_tickers=None, progress_callback=None, use_multiprocess=True):
        """
        批量並行預測 v2.0
        
        Args:
            ticker_list: 股票列表
            priority_tickers: 優先處理的股票
            progress_callback: 進度回調 func(current, total)
            use_multiprocess: True=多進程(快), False=多線程(兼容)
        
        Returns:
            list: 預測結果列表
        """
        # 優先級排序
        if priority_tickers:
            priority_set = set(priority_tickers)
            sorted_tickers = (
                [t for t in ticker_list if t in priority_set] +
                [t for t in ticker_list if t not in priority_set]
            )
        else:
            sorted_tickers = ticker_list
        
        results = []
        total = len(sorted_tickers)
        
        if use_multiprocess and total > 10:
            print(f"🚀 使用 {self.max_workers} 個進程並行處理 {total} 檔股票 (多進程模式，CPU 將滿載)...")
            
            # 多進程並行 (繞過 GIL)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任務
                future_to_ticker = {
                    executor.submit(_predict_worker, ticker, self.ai_service): ticker
                    for ticker in sorted_tickers
                }
                
                # 收集結果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result(timeout=60)
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"處理 {ticker} 異常: {e}")
                    
                    completed += 1
                    
                    # 進度回報
                    if progress_callback:
                        progress_callback(completed, total)
                    elif completed % 5 == 0:
                        print(f"進度: {completed}/{total} ({completed/total*100:.1f}%)")
        else:
            print(f"🚀 使用 {self.max_workers} 線程並行處理 {total} 檔股票...")
            
            # 多線程模式 (小批量或兼容模式)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._predict_single_stock, ticker): ticker
                    for ticker in sorted_tickers
                }
                
                completed = 0
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result(timeout=30)
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"處理 {ticker} 異常: {e}")
                    
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
                    elif completed % 10 == 0:
                        print(f"進度: {completed}/{total} ({completed/total*100:.1f}%)")
        
        return results
    
    def clear_cache(self):
        """清除快取"""
        self.cache.clear()


# 全域函數供多進程使用 (必須在模組頂層)
def _predict_worker(ticker, ai_service_class=None):
    """
    多進程 worker 函數
    
    注意: 多進程無法共享 ai_service 物件，需重新創建
    """
    try:
        from services.ai_service import AIService
        
        # 在子進程中創建獨立的 AI Service
        ai_service = AIService()
        
        # 獲取數據
        symbol, hist_data, current_price, _ = get_stock_data(ticker)
        
        if hist_data.empty:
            return None
        
        # 執行預測
        pred_price = ai_service.predict_next_price(ticker, hist_data)
        
        if not pred_price:
            return None
        
        # 計算基準價格
        today = datetime.datetime.now().date()
        hist_data.index = pd.to_datetime(hist_data.index)
        before_today = hist_data[hist_data.index.date < today]
        
        if not before_today.empty:
            base_price = float(before_today['Close'].iloc[-1])
        else:
            base_price = float(hist_data['Close'].iloc[-1])
        
        expected_change = ((pred_price - base_price) / base_price) * 100
        
        return {
            "ticker": ticker,
            "base_price": base_price,
            "predicted_price": float(pred_price),
            "expected_change_pct": round(expected_change, 2)
        }
    
    except Exception as e:
        print(f"Worker 預測 {ticker} 失敗: {e}")
        return None


def optimized_morning_predictions(ai_service, max_stocks=None, priority_tickers=None):
    """
    優化版早盤預測
    
    用法:
        from services.ai_prediction_speedup import optimized_morning_predictions
        count = optimized_morning_predictions(ai_service, max_stocks=50)
    
    Args:
        ai_service: AIService 實例
        max_stocks: 最多預測幾檔 (None = 全部)
        priority_tickers: 優先處理的股票列表
    
    Returns:
        int: 成功預測數量
    """
    start_time = time.time()
    
    # 取得所有已訓練模型
    ticker_list = list(ai_service.models.keys())
    
    if max_stocks and len(ticker_list) > max_stocks:
        # 限制數量 (優先處理持倉股票)
        if priority_tickers:
            priority_set = set(priority_tickers)
            selected = (
                [t for t in ticker_list if t in priority_set][:max_stocks//2] +
                [t for t in ticker_list if t not in priority_set][:max_stocks//2]
            )
            ticker_list = selected[:max_stocks]
        else:
            ticker_list = ticker_list[:max_stocks]
    
    print(f"📊 開始預測 {len(ticker_list)} 檔股票...")
    
    # 創建快速預測引擎
    engine = FastPredictionEngine(ai_service)
    
    # 執行批量預測
    results = engine.batch_predict(
        ticker_list,
        priority_tickers=priority_tickers,
        progress_callback=lambda c, t: None
    )
    
    # 保存結果
    today_str = datetime.datetime.now().date().strftime("%Y-%m-%d")
    
    # 加上日期標記
    for result in results:
        result['date'] = today_str
        result['actual_price'] = None
        result['diff_pct'] = None
        result['accuracy'] = "等待收盤驗證"
    
    # 合併到舊記錄
    prediction_file = "data/daily_predictions.json"
    all_records = []
    
    if os.path.exists(prediction_file):
        try:
            with open(prediction_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        except:
            pass
    
    # 移除今天的舊記錄
    all_records = [r for r in all_records if r['date'] != today_str]
    all_records.extend(results)
    
    # 保存
    with open(prediction_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=4)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ 預測完成!")
    print(f"   成功: {len(results)}/{len(ticker_list)} 檔")
    print(f"   耗時: {elapsed:.1f} 秒 (平均 {elapsed/len(ticker_list):.2f} 秒/檔)")
    
    return len(results)


def quick_predict_portfolio(ai_service, portfolio_tickers):
    """
    快速預測持倉股票 (最優先) v2.0
    
    用於對戰系統中的即時決策
    
    Args:
        ai_service: AIService 實例
        portfolio_tickers: 持倉股票列表
    
    Returns:
        dict: {ticker: prediction_result}
    """
    print(f"⚡ 快速預測 {len(portfolio_tickers)} 檔持倉股票 (多進程模式)...")
    
    engine = FastPredictionEngine(ai_service)
    # 持倉股票用多進程，充分利用 CPU
    results = engine.batch_predict(portfolio_tickers, use_multiprocess=True)
    
    # 轉換為字典
    predictions = {r['ticker']: r for r in results if r}
    
    return predictions


# 集成到 Streamlit UI 的進度條版本
def predict_with_progress_bar(ai_service, ticker_list, st_progress_bar=None, st_status_text=None):
    """
    帶進度條的預測 (用於 Streamlit UI)
    
    用法:
        progress_bar = st.progress(0)
        status_text = st.empty()
        count = predict_with_progress_bar(
            ai_service, 
            ticker_list, 
            progress_bar, 
            status_text
        )
    """
    engine = FastPredictionEngine(ai_service)
    total = len(ticker_list)
    
    def update_progress(current, total):
        if st_progress_bar:
            st_progress_bar.progress(current / total)
        if st_status_text:
            st_status_text.text(f"預測進度: {current}/{total} ({current/total*100:.0f}%)")
    
    results = engine.batch_predict(
        ticker_list,
        progress_callback=update_progress
    )
    
    # 保存結果
    today_str = datetime.datetime.now().date().strftime("%Y-%m-%d")
    
    for result in results:
        result['date'] = today_str
        result['actual_price'] = None
        result['diff_pct'] = None
        result['accuracy'] = "等待收盤驗證"
    
    prediction_file = "data/daily_predictions.json"
    all_records = []
    
    if os.path.exists(prediction_file):
        try:
            with open(prediction_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        except:
            pass
    
    all_records = [r for r in all_records if r['date'] != today_str]
    all_records.extend(results)
    
    with open(prediction_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=4)
    
    return len(results)
