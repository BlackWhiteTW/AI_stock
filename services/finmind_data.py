"""
FinMind 財經數據整合
提供完整的台股歷史財務數據

官方網站: https://finmindtrade.com/
GitHub: https://github.com/FinMind/FinMind
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path


class FinMindDataFetcher:
    """
    FinMind 數據抓取器
    提供台股完整的歷史財務報表、股價、營收等數據
    數據以 CSV 格式分別存儲，便於閱讀和分析
    """
    
    API_URL = "https://api.finmindtrade.com/api/v4/data"
    
    def __init__(self, api_token=None, base_dir='data/financial_data'):
        """
        初始化 FinMind 抓取器
        :param api_token: API Token（免費版可以不提供，但有請求限制）
        :param base_dir: 數據存儲根目錄（不再使用 cache 子資料夾）
        """
        self.api_token = api_token
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_request(self, dataset, **params):
        """
        發送 API 請求
        :param dataset: 數據集名稱
        :param params: 查詢參數
        :return: dict API 回應
        """
        params['dataset'] = dataset
        
        if self.api_token:
            params['token'] = self.api_token
        
        try:
            response = requests.get(self.API_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            print(f"❌ API 請求失敗: {e}")
            return None
    
    def get_financial_statements(self, stock_id, start_date, end_date=None):
        """
        獲取財務報表（季度）
        數據以 CSV 格式存儲，易於閱讀和分析
        
        :param stock_id: 股票代碼（不含 .TW，如 '2330'）
        :param start_date: 開始日期 'YYYY-MM-DD'
        :param end_date: 結束日期 'YYYY-MM-DD'（預設為今天）
        :return: DataFrame 財務報表數據
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        # 建立股票專屬目錄
        stock_dir = self.base_dir / f"{stock_id}.TW"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 檔案路徑（使用年份範圍作為檔名）
        start_year = start_date[:4]
        end_year = end_date[:4]
        csv_file = stock_dir / f"financial_statements_{start_year}_{end_year}.csv"
        
        # 檢查快取（7天內有效）
        if csv_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
            if cache_age < timedelta(days=7):
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"📂 從快取讀取: {csv_file.name}")
                    return df
                except:
                    pass
        
        print(f"📊 正在獲取 {stock_id} 財務報表 ({start_date} ~ {end_date})")
        
        response = self._make_request(
            dataset='TaiwanStockFinancialStatements',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if response and response.get('status') == 200:
            data = response.get('data', [])
            df = pd.DataFrame(data)
            
            if not df.empty:
                # 儲存為 CSV（使用 utf-8-sig 以支援 Excel 直接開啟）
                try:
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    print(f"✅ 成功獲取 {len(df)} 筆 → 已存至 {csv_file}")
                except Exception as e:
                    print(f"⚠️ 儲存 CSV 失敗: {e}")
                
                return df
        
        print(f"❌ 無法獲取財務報表數據")
        return pd.DataFrame()
    
    def get_balance_sheet(self, stock_id, start_date, end_date=None):
        """
        獲取資產負債表
        數據以 CSV 格式存儲
        
        :param stock_id: 股票代碼
        :param start_date: 開始日期
        :param end_date: 結束日期
        :return: DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        # 建立股票專屬目錄
        stock_dir = self.base_dir / f"{stock_id}.TW"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        start_year = start_date[:4]
        end_year = end_date[:4]
        csv_file = stock_dir / f"balance_sheet_{start_year}_{end_year}.csv"
        
        # 檢查快取
        if csv_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
            if cache_age < timedelta(days=7):
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"📂 從快取讀取: {csv_file.name}")
                    return df
                except:
                    pass
        
        print(f"📋 正在獲取 {stock_id} 資產負債表 ({start_date} ~ {end_date})")
        
        response = self._make_request(
            dataset='TaiwanStockBalanceSheet',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if response and response.get('status') == 200:
            df = pd.DataFrame(response.get('data', []))
            if not df.empty:
                try:
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    print(f"✅ 成功獲取 {len(df)} 筆 → 已存至 {csv_file}")
                except Exception as e:
                    print(f"⚠️ 儲存 CSV 失敗: {e}")
                return df
        
        return pd.DataFrame()
    
    def get_cash_flow(self, stock_id, start_date, end_date=None):
        """
        獲取現金流量表
        數據以 CSV 格式存儲
        
        :param stock_id: 股票代碼
        :param start_date: 開始日期
        :param end_date: 結束日期
        :return: DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        # 建立股票專屬目錄
        stock_dir = self.base_dir / f"{stock_id}.TW"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        start_year = start_date[:4]
        end_year = end_date[:4]
        csv_file = stock_dir / f"cash_flow_{start_year}_{end_year}.csv"
        
        # 檢查快取
        if csv_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
            if cache_age < timedelta(days=7):
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"📂 從快取讀取: {csv_file.name}")
                    return df
                except:
                    pass
        
        print(f"💰 正在獲取 {stock_id} 現金流量表 ({start_date} ~ {end_date})")
        
        response = self._make_request(
            dataset='TaiwanStockCashFlowsStatement',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if response and response.get('status') == 200:
            df = pd.DataFrame(response.get('data', []))
            if not df.empty:
                try:
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    print(f"✅ 成功獲取 {len(df)} 筆 → 已存至 {csv_file}")
                except Exception as e:
                    print(f"⚠️ 儲存 CSV 失敗: {e}")
                return df
        
        return pd.DataFrame()
    
    def get_monthly_revenue(self, stock_id, start_date, end_date=None):
        """
        獲取月營收數據
        數據以 CSV 格式存儲
        
        :param stock_id: 股票代碼
        :param start_date: 開始日期
        :param end_date: 結束日期
        :return: DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        # 建立股票專屬目錄
        stock_dir = self.base_dir / f"{stock_id}.TW"
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        start_year = start_date[:4]
        end_year = end_date[:4]
        csv_file = stock_dir / f"monthly_revenue_{start_year}_{end_year}.csv"
        
        # 檢查快取
        if csv_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(csv_file.stat().st_mtime)
            if cache_age < timedelta(days=7):
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    print(f"📂 從快取讀取: {csv_file.name}")
                    return df
                except:
                    pass
        
        print(f"📈 正在獲取 {stock_id} 月營收 ({start_date} ~ {end_date})")
        
        response = self._make_request(
            dataset='TaiwanStockMonthRevenue',
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if response and response.get('status') == 200:
            df = pd.DataFrame(response.get('data', []))
            if not df.empty:
                try:
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    print(f"✅ 成功獲取 {len(df)} 筆 → 已存至 {csv_file}")
                except Exception as e:
                    print(f"⚠️ 儲存 CSV 失敗: {e}")
                return df
        
        return pd.DataFrame()
    
    def get_all_historical_data(self, stock_id, start_date, end_date=None):
        """
        一次獲取所有歷史財務數據
        
        :param stock_id: 股票代碼
        :param start_date: 開始日期 'YYYY-MM-DD'
        :param end_date: 結束日期
        :return: dict 包含所有財務數據
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'='*70}")
        print(f"📊 開始獲取 {stock_id} 完整歷史財務數據")
        print(f"時間範圍: {start_date} ~ {end_date}")
        print(f"{'='*70}\n")
        
        result = {
            'stock_id': stock_id,
            'start_date': start_date,
            'end_date': end_date,
            'fetch_time': datetime.now().isoformat(),
            'financial_statements': None,
            'balance_sheet': None,
            'cash_flow': None,
            'monthly_revenue': None
        }
        
        # 1. 損益表
        result['financial_statements'] = self.get_financial_statements(stock_id, start_date, end_date)
        
        # 2. 資產負債表
        result['balance_sheet'] = self.get_balance_sheet(stock_id, start_date, end_date)
        
        # 3. 現金流量表
        result['cash_flow'] = self.get_cash_flow(stock_id, start_date, end_date)
        
        # 4. 月營收
        result['monthly_revenue'] = self.get_monthly_revenue(stock_id, start_date, end_date)
        
        print(f"\n{'='*70}")
        print(f"✨ 完成！數據摘要:")
        print(f"  - 財務報表: {len(result['financial_statements'])} 筆")
        print(f"  - 資產負債表: {len(result['balance_sheet'])} 筆")
        print(f"  - 現金流量表: {len(result['cash_flow'])} 筆")
        print(f"  - 月營收: {len(result['monthly_revenue'])} 筆")
        print(f"{'='*70}\n")
        
        return result


# 全局實例
_finmind_fetcher = None

def get_finmind_fetcher(api_token=None):
    """獲取 FinMind 抓取器單例"""
    global _finmind_fetcher
    if _finmind_fetcher is None:
        _finmind_fetcher = FinMindDataFetcher(api_token=api_token)
    return _finmind_fetcher


# 使用範例
if __name__ == '__main__':
    fetcher = FinMindDataFetcher()
    
    # 範例 1: 獲取台積電 2020-2024 的所有財務數據
    print("範例: 獲取台積電 5 年歷史財務數據")
    
    result = fetcher.get_all_historical_data(
        stock_id='2330',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # 顯示財務報表前幾筆
    if result['financial_statements'] is not None and not result['financial_statements'].empty:
        df = result['financial_statements']
        print("\n財務報表欄位:")
        print(df.columns.tolist())
        print("\n前 5 筆數據:")
        print(df.head())
    
    # 顯示月營收前幾筆
    if result['monthly_revenue'] is not None and not result['monthly_revenue'].empty:
        df = result['monthly_revenue']
        print("\n月營收欄位:")
        print(df.columns.tolist())
        print("\n前 5 筆數據:")
        print(df.head())
