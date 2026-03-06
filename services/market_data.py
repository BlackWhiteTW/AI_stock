import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class MarketDataManager:
    def __init__(self, data_dir='data/stock_data'):
        """
        初始化數據管理器
        :param data_dir: 存放 CSV 檔案的資料夾路徑
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_data(self, ticker, period="max", only_cached=False):
        """
        載入股票數據：
        1. 檢查本地是否有存檔
        2. 如果有，檢查是否需要更新 (差異更新)
        3. 如果無，下載完整歷史數據
        4. 存檔並回傳 DataFrame
        :param only_cached: 若為 True，則只讀取本地檔案，不嘗試聯網更新 (提升批量處理速度)
        """
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")
        ticker_obj = yf.Ticker(ticker)
        
        df = pd.DataFrame()
        needs_save = False
        
        # 1. 嘗試讀取本地檔案
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                # 處理 Timezone 問題，統一轉為 None (Naive)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                if not df.empty and not only_cached:
                    last_date = df.index[-1].date()
                    today = datetime.now().date()
                    
                    # 如果最後一筆數據不是今天 (且今天還沒過完，或者昨天收盤數據沒抓到)
                    # 簡單判斷：若最後日期 < 今天日期，嘗試更新
                    if last_date < today:
                        # 為了保險，我們抓取 last_date 之後的數據
                        # yfinance 的 start 是 inclusive，所以我們+1天
                        start_date = last_date + timedelta(days=1)
                        
                        # 如果 start_date 就是今天，且現在是盤中，可能會有數據
                        # 若 start_date > today (未來)，則不抓
                        if start_date <= today:
                            # 抓取新數據
                            # print(f"[{ticker}] 正在更新數據... 從 {start_date} 開始")
                            new_data = ticker_obj.history(start=start_date)
                            
                            if not new_data.empty:
                                # 處理 Timezone
                                if new_data.index.tz is not None:
                                    new_data.index = new_data.index.tz_localize(None)
                                
                                # 再次過濾，確保沒有重疊
                                new_data = new_data[new_data.index.date > last_date]
                                
                                if not new_data.empty:
                                    df = pd.concat([df, new_data])
                                    needs_save = True
            except Exception as e:
                print(f"讀取或更新本地數據失敗: {e}，將重新下載完整數據。")
                df = pd.DataFrame() # 重置，觸發下方完整下載

        # 2. 如果本地無數據或讀取失敗，下載完整數據
        if df.empty:
            try:
                # [強力靜音模式] 關閉所有 yfinance 相關日誌與錯誤輸出
                import logging
                for logger_name in ['yfinance', 'peewee', 'urllib3']:
                    logger = logging.getLogger(logger_name)
                    logger.setLevel(logging.CRITICAL)
                    logger.propagate = False
                
                # 這裡使用 period="max" 確保抓到所有歷史
                try:
                    # 加上 progress=False 禁止進度條干擾
                    df = ticker_obj.history(period="10y", progress=False) 
                    if df.empty:
                        df = ticker_obj.history(period="max", progress=False)
                except:
                    pass
                    
            except Exception as e:
                pass 
                
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                needs_save = True
        
        # 3. 存檔
        if needs_save and not df.empty:
            try:
                df.to_csv(file_path)
            except:
                pass # 存檔失敗忽略
            
        return df

    def consolidate_to_parquet(self, output_path='data/market_snapshot.parquet'):
        """
        🚀 高性能優化：將所有分散的 CSV 整合為單一 Parquet 檔案
        這能極大提升 GNN 與全市場掃描的 I/O 速度
        """
        import glob
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not all_files:
            return None
            
        print(f"📦 正在整合 {len(all_files)} 個數據源至 Parquet 快照...")
        
        all_df = []
        for f in all_files:
            try:
                ticker = os.path.basename(f).replace(".csv", "")
                df = pd.read_csv(f)
                df['Ticker'] = ticker
                # 確保 Date 是 datetime 格式
                df['Date'] = pd.to_datetime(df['Date'])
                all_df.append(df)
            except:
                continue
                
        if not all_df:
            return None
            
        # 合併大表
        full_df = pd.concat(all_df, ignore_index=True)
        
        # 存儲為 Parquet (使用 Snappy 壓縮)
        full_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        print(f"✅ 市場快照已儲存: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
        return output_path

    def load_market_snapshot(self, snapshot_path='data/market_snapshot.parquet'):
        """讀取整合後的市場快照"""
        if os.path.exists(snapshot_path):
            try:
                return pd.read_parquet(snapshot_path, engine='pyarrow')
            except Exception as e:
                print(f"讀取 Parquet 失敗: {e}")
        return None

    def get_data_range(self, ticker, days=None):
        """
        取得特定範圍的數據
        :param days: 回傳最近 N 天的數據，None 則回傳全部
        """
        df = self.load_data(ticker)
        
        if df.empty:
            return df
            
        if days:
            # 確保不 truncate 超過有的數據
            if len(df) > days:
                return df.iloc[-days:]
        
        return df
