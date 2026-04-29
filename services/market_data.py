"""
🚀 混合動力市場數據管理器 (Hybrid Market Data Manager)
整合 00-ff 雜湊快取與產業聚合快照，實現併行共存與極限效能
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from .cache_manager import get_cache_manager # 🚀 對接專案統一快取系統

class MarketDataManager:
    def __init__(self, data_dir='data/stock_data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 取得統一快取管理器實例
        self.cache_manager = get_cache_manager()
        # 定義聚合快照專屬路徑
        self.industry_snapshot_dir = 'data/unified_cache/industry_snapshots'
        os.makedirs(self.industry_snapshot_dir, exist_ok=True)

    def load_data(self, ticker, period="max", only_cached=False):
        """
        載入股票數據：優先從 00-ff 統一快取讀取
        """
        print(f"📊 [MarketData] 正在載入 {ticker} ...")
        # 1. 嘗試從統一快取系統獲取 (00-ff 邏輯)
        try:
            cached_df = self.cache_manager.get("stock_data", ticker)
        except Exception as e:
            print(f"⚠️ [MarketData] 快取讀取失敗: {e}")
            cached_df = None
        
        if cached_df is not None and isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
            if not only_cached:
                # 確保索引是日期時間格式且無時區
                if cached_df.index.tz is not None:
                    cached_df.index = cached_df.index.tz_localize(None)
                
                # 檢查更新邏輯
                last_date = cached_df.index[-1].date()
                today = datetime.now().date()
                
                # 🚀 優化：如果是今天但成交量為 0 (盤前數據)，或日期落後，則更新
                needs_update = last_date < today
                if not needs_update and last_date == today:
                    if 'Volume' in cached_df.columns and (cached_df['Volume'].iloc[-1] == 0 or pd.isna(cached_df['Volume'].iloc[-1])):
                        needs_update = True
                        print(f"🔄 [MarketData] 偵測到 {ticker} 今日數據不完整 (Vol=0)，嘗試重新獲取實時數據...")

                if needs_update:
                    # 如果是今天且成交量為 0，fetch_start 應設為今天以覆蓋，否則設為最後一天的隔天
                    fetch_start = last_date if (last_date == today) else (last_date + timedelta(days=1))
                    print(f"🔄 [MarketData] {ticker} 數據需要更新 (最後日期: {last_date}, 目標: {today})...")
                    
                    new_data = self._fetch_new_data(ticker, fetch_start)
                    if new_data is not None and not new_data.empty:
                        # 合併並去重
                        cached_df = pd.concat([cached_df, new_data])
                        # 按索引排序並去重，保留最新的一筆
                        cached_df.sort_index(inplace=True)
                        cached_df = cached_df[~cached_df.index.duplicated(keep='last')]
                        
                        self.cache_manager.set("stock_data", ticker, cached_df, ttl=30*24*3600)
                        print(f"✅ [MarketData] {ticker} 數據更新成功 ({len(new_data)} 筆新數據)")
                    else:
                        print(f"ℹ️ [MarketData] {ticker} 獲取新數據失敗或無新數據")
            return cached_df

        if only_cached: 
            print(f"ℹ️ [MarketData] {ticker} 無快取且處於僅快取模式")
            return pd.DataFrame()

        # 2. 下載新數據
        print(f"🌐 [MarketData] 正在從 Yahoo Finance 下載 {ticker} 完整數據...")
        df = self._download_full_data(ticker, period)
        if not df.empty:
            self.cache_manager.set("stock_data", ticker, df, ttl=30*24*3600)
            # 備份到 CSV
            self._backup_to_csv(ticker, df)
            print(f"✅ [MarketData] {ticker} 下載並存儲完成")
        else:
            print(f"❌ [MarketData] {ticker} 下載失敗")
        return df

    def _fetch_new_data(self, ticker, start_date):
        """從 Yahoo Finance 獲取新數據 (增強型)"""
        try:
            ticker_obj = yf.Ticker(ticker)
            # start 參數接受 string, datetime 或 date
            start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
            
            # 🚀 嘗試獲取數據
            new_data = ticker_obj.history(start=start_str)
            
            if not new_data.empty:
                # 強制統一格式：無時區
                if new_data.index.tz is not None:
                    new_data.index = new_data.index.tz_localize(None)
                
                # 再次過濾確保日期符合 (yfinance 有時會多給一天)
                if hasattr(start_date, 'year'):
                    target_ts = pd.Timestamp(start_date)
                    new_data = new_data[new_data.index >= target_ts]
                    
                return new_data
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ [MarketData] _fetch_new_data 發生錯誤: {e}")
            return None

    def _download_full_data(self, ticker, period="max"):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="10y")
            if df.empty: df = ticker_obj.history(period="max")
            if not df.empty and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except: return pd.DataFrame()

    def _backup_to_csv(self, ticker, df):
        try:
            file_path = os.path.join(self.data_dir, f"{ticker}.csv")
            df.to_csv(file_path)
        except: pass

    def maintain_categorized_snapshots(self, force=False):
        """
        🚀 產業聚合功能：從 00-ff 快取提取數據，併行生成產業大 Parquet
        """
        from utils import get_tw_stock_list
        df_stocks = get_tw_stock_list()
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        status_file = os.path.join(self.industry_snapshot_dir, "last_update.txt")
        if not force and os.path.exists(status_file):
            with open(status_file, 'r') as f:
                if f.read().strip() == today_str:
                    return True

        print(f"📦 正在從 00-ff 快取生成產業聚合快照...")
        all_data = []
        for _, row in df_stocks.iterrows():
            ticker = row['Ticker']
            industry = row['產業']
            # 從統一快取讀取單股 (超快)
            df = self.cache_manager.get("stock_data", ticker)
            if df is not None and not df.empty:
                df = df.copy()
                df['Ticker'] = ticker
                df['Industry'] = industry
                all_data.append(df.reset_index())
        
        if not all_data: return False

        full_df = pd.concat(all_data, ignore_index=True)
        for industry, group in full_df.groupby('Industry'):
            safe_name = str(industry).replace("/", "_").replace("\\", "_")
            file_path = os.path.join(self.industry_snapshot_dir, f"{safe_name}.parquet")
            group.to_parquet(file_path, engine='pyarrow', compression='snappy')

        with open(status_file, 'w') as f:
            f.write(today_str)
        print(f"✅ 併行快照同步完成：產業聚合檔案存於 {self.industry_snapshot_dir}")
        return True

    def load_industry_snapshot(self, industry):
        """從專屬聚合目錄讀取產業數據"""
        safe_name = str(industry).replace("/", "_").replace("\\", "_")
        file_path = os.path.join(self.industry_snapshot_dir, f"{safe_name}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            return df
        return pd.DataFrame()

    def consolidate_to_parquet(self, output_path='data/market_snapshot.parquet'):
        """
        🚀 高性能優化：將所有分散的 CSV 整合為單一 Parquet 檔案
        (主模型訓練專用)
        """
        import glob
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not all_files: return None
            
        print(f"📦 正在整合 {len(all_files)} 個數據源至 Parquet 快照...")
        all_df = []
        for f in all_files:
            try:
                ticker = os.path.basename(f).replace(".csv", "")
                df = pd.read_csv(f)
                df['Ticker'] = ticker
                df['Date'] = pd.to_datetime(df['Date'])
                all_df.append(df)
            except: continue
                
        if not all_df: return None
        full_df = pd.concat(all_df, ignore_index=True)
        full_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        return output_path

    def load_market_snapshot(self, snapshot_path='data/market_snapshot.parquet'):
        """讀取整合後的市場大快照 (主模型訓練專用)"""
        if os.path.exists(snapshot_path):
            try:
                return pd.read_parquet(snapshot_path, engine='pyarrow')
            except Exception as e:
                print(f"讀取 Parquet 失敗: {e}")
        return None
