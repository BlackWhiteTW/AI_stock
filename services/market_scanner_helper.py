"""
Market Scanner Helper
市場掃描輔助模塊 - 提供統一的市場掃描功能
"""
import streamlit as st
import pandas as pd
import random
from typing import List, Tuple, Dict, Optional, Callable


class MarketScannerUI:
    """市場掃描 UI 輔助類 - 統一的進度反饋和錯誤處理"""
    
    def __init__(self, ai_advisor):
        self.ai_advisor = ai_advisor
        self.last_scan_time = 0
        self.last_results = []
    
    def scan_with_progress(
        self,
        tickers: List[str],
        include_news: bool = False,
        use_cache: bool = True,
        min_score: int = 0,
        use_parallel: bool = True,
        max_workers: int = 6
    ) -> Tuple[List[Dict], float]:
        """
        執行市場掃描並顯示進度
        
        Args:
            tickers: 股票代碼列表
            include_news: 是否包含新聞分析
            use_cache: 是否使用快取
            min_score: 最低分數篩選
            use_parallel: 是否使用並行處理
            max_workers: 最大並行數
            
        Returns:
            (scan_results, scan_time)
        """
        import time
        from utils import get_stock_data
        
        start_time = time.time()
        results = []
        
        # 建立進度顯示
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_text = st.empty()
        
        try:
            if use_parallel:
                # 使用並行掃描引擎
                results = self._parallel_scan(
                    tickers, include_news, use_cache, min_score, 
                    max_workers, progress_bar, status_text, stats_text
                )
            else:
                # 使用順序掃描
                results = self._sequential_scan(
                    tickers, include_news, min_score,
                    progress_bar, status_text, stats_text
                )
        
        except Exception as e:
            st.error(f"掃描過程發生錯誤: {e}")
            st.warning("嘗試使用備用掃描方式...")
            
            # Fallback 到順序掃描
            results = self._sequential_scan(
                tickers, include_news, min_score,
                progress_bar, status_text, stats_text
            )
        
        finally:
            # 清除進度顯示
            progress_bar.empty()
            status_text.empty()
            stats_text.empty()
        
        scan_time = time.time() - start_time
        self.last_scan_time = scan_time
        self.last_results = results
        
        return results, scan_time
    
    def _parallel_scan(
        self, tickers, include_news, use_cache, min_score,
        max_workers, progress_bar, status_text, stats_text
    ):
        """並行掃描實現"""
        from services.performance_optimizer import get_stock_scanner
        from utils import get_stock_data
        
        scanner = get_stock_scanner(self.ai_advisor, max_workers=max_workers)
        
        # 步驟 1: 下載數據
        status_text.text("📥 正在批次下載股票數據...")
        stock_data_list = []
        
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers) * 0.4)  # 下載佔 40%
            try:
                symbol, hist, price, info = get_stock_data(ticker)
                if len(hist) > 30:
                    stock_data_list.append((symbol, hist))
            except Exception as e:
                continue
        
        stats_text.text(f"✅ 已載入 {len(stock_data_list)}/{len(tickers)} 檔股票數據")
        
        # 步驟 2: 並行分析
        status_text.text(f"🧠 AI 正在並行分析 {len(stock_data_list)} 檔股票...")
        
        analysis_results = scanner.scan_stocks_parallel(
            stock_data_list,
            include_news=include_news,
            use_cache=use_cache
        )
        
        progress_bar.progress(1.0)
        
        # 整理結果
        scan_results = []
        for ticker, analysis in analysis_results.items():
            if analysis and analysis.get('score', 0) >= min_score:
                scan_results.append({
                    '股票': ticker,
                    '建議': analysis.get('action', '未知'),
                    '分數': analysis.get('score', 0),
                    '理由': ', '.join(analysis.get('reasons', [])[:3])
                })
        
        return scan_results
    
    def _sequential_scan(
        self, tickers, include_news, min_score,
        progress_bar, status_text, stats_text
    ):
        """順序掃描實現（備用方式）"""
        from utils import get_stock_data
        
        scan_results = []
        found_count = 0
        
        for i, ticker in enumerate(tickers):
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"分析中... {ticker} ({i+1}/{len(tickers)})")
            stats_text.text(f"已找到 {found_count} 檔符合條件的股票")
            
            try:
                symbol, hist, price, info = get_stock_data(ticker)
                if len(hist) > 30:
                    analysis = self.ai_advisor.analyze_stock(
                        symbol, hist, include_news=include_news
                    )
                    
                    if analysis['score'] >= min_score:
                        found_count += 1
                        scan_results.append({
                            '股票': symbol,
                            '建議': analysis['action'],
                            '分數': analysis['score'],
                            '理由': ', '.join(analysis['reasons'][:3])
                        })
            except Exception as e:
                continue
        
        return scan_results
    
    def display_results(self, results: List[Dict], scan_time: float):
        """顯示掃描結果"""
        if results:
            st.success(f"✅ 找到 {len(results)} 檔具有潛力的股票！")
            
            df_results = pd.DataFrame(results).sort_values('分數', ascending=False)
            st.dataframe(df_results, use_container_width=True)
            
            # 顯示統計
            avg_score = df_results['分數'].mean()
            max_score = df_results['分數'].max()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("總數", len(results))
            col2.metric("平均分數", f"{avg_score:.1f}")
            col3.metric("最高分數", f"{max_score:.1f}")
            
            st.caption(f"⚡ 掃描耗時: {scan_time:.1f} 秒")
        else:
            st.info("目前沒有發現明顯的買入訊號。")


class SmartTickerSelector:
    """智能股票選擇器 - 用於選擇掃描目標"""
    
    @staticmethod
    def select_by_industry(
        df_stocks: pd.DataFrame,
        samples_per_industry: int = 3,
        max_total: int = 100
    ) -> List[str]:
        """
        按產業分類隨機抽樣
        
        Args:
            df_stocks: 股票清單 DataFrame
            samples_per_industry: 每個產業抽樣數量
            max_total: 總數上限
            
        Returns:
            股票代碼列表
        """
        if df_stocks.empty or '產業' not in df_stocks.columns:
            return []
        
        selected = []
        industries = df_stocks['產業'].unique()
        
        for industry in industries:
            stocks_in_industry = df_stocks[
                df_stocks['產業'] == industry
            ]['Ticker'].tolist()
            
            if stocks_in_industry:
                sample_size = min(samples_per_industry, len(stocks_in_industry))
                sampled = random.sample(stocks_in_industry, sample_size)
                selected.extend(sampled)
        
        # 去重並限制總數
        selected = list(set(selected))
        if len(selected) > max_total:
            selected = random.sample(selected, max_total)
        
        return selected
    
    @staticmethod
    def merge_with_watchlist(
        base_tickers: List[str],
        watchlist: Dict,
        portfolio: Dict = None
    ) -> List[str]:
        """
        合併基礎清單、自選清單和持倉
        
        Args:
            base_tickers: 基礎股票列表
            watchlist: 自選清單字典
            portfolio: 持倉字典
            
        Returns:
            合併後的股票代碼列表
        """
        all_tickers = set(base_tickers)
        
        # 添加自選清單
        if watchlist:
            for ticker in watchlist.keys():
                # 標準化代碼
                if not (ticker.endswith('.TW') or ticker.endswith('.TWO')):
                    ticker = ticker + '.TW'
                all_tickers.add(ticker)
        
        # 添加持倉
        if portfolio:
            for ticker in portfolio.keys():
                all_tickers.add(ticker)
        
        return list(all_tickers)


def create_scan_mode_selector() -> Tuple[bool, bool]:
    """
    創建掃描模式選擇器
    
    Returns:
        (include_news, use_cache)
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_mode = st.radio(
            "掃描模式",
            ["⚡ 快速掃描（僅技術面）", "🌐 深度掃描（含新聞分析）"],
            horizontal=True
        )
        include_news = "深度掃描" in scan_mode
    
    with col2:
        use_cache = st.checkbox("使用快取", value=True, help="使用快取可大幅提升速度")
    
    return include_news, use_cache
