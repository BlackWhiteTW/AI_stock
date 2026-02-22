"""
台灣證交所公開資訊觀測站 (MOPS) 數據抓取器
可獲取完整的歷史財務報表（可追溯多年）

官方網站: https://mops.twse.com.tw
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import io


class MOPSDataFetcher:
    """
    MOPS 公開資訊觀測站數據抓取器
    支援獲取完整的歷史財務報表
    """
    
    BASE_URL = "https://mops.twse.com.tw"
    
    def __init__(self, cache_dir='data/mops_cache'):
        """
        初始化 MOPS 抓取器
        :param cache_dir: 快取目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定 User-Agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_financial_statement(self, stock_id, year, season, report_type='consolidated'):
        """
        獲取單季財務報表
        
        :param stock_id: 股票代碼（不含 .TW，如 '2330'）
        :param year: 民國年（如 113 代表 2024 年）
        :param season: 季度 (1-4)
        :param report_type: 'consolidated' 合併報表 或 'individual' 個別報表
        :return: dict 包含三大報表數據
        """
        # 清理股票代碼
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        # 檢查快取
        cache_file = self.cache_dir / f"{stock_id}_{year}Q{season}_{report_type}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        result = {
            'stock_id': stock_id,
            'year': year,
            'season': season,
            'report_type': report_type,
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'fetch_time': datetime.now().isoformat()
        }
        
        # 1. 綜合損益表
        try:
            income_url = f"{self.BASE_URL}/mops/web/ajax_t163sb04"
            income_data = {
                'encodeURIComponent': '1',
                'step': '1',
                'firstin': '1',
                'off': '1',
                'co_id': stock_id,
                'year': str(year),
                'season': str(season).zfill(2)
            }
            
            response = requests.post(income_url, data=income_data, headers=self.headers, timeout=10)
            if response.status_code == 200:
                result['income_statement'] = self._parse_html_table(response.text)
                print(f"✅ 已獲取 {stock_id} {year}Q{season} 損益表")
            
            time.sleep(3)  # 避免請求過快
            
        except Exception as e:
            print(f"獲取損益表失敗: {e}")
        
        # 2. 資產負債表
        try:
            balance_url = f"{self.BASE_URL}/mops/web/ajax_t163sb05"
            balance_data = {
                'encodeURIComponent': '1',
                'step': '1',
                'firstin': '1',
                'off': '1',
                'co_id': stock_id,
                'year': str(year),
                'season': str(season).zfill(2)
            }
            
            response = requests.post(balance_url, data=balance_data, headers=self.headers, timeout=10)
            if response.status_code == 200:
                result['balance_sheet'] = self._parse_html_table(response.text)
                print(f"✅ 已獲取 {stock_id} {year}Q{season} 資產負債表")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"獲取資產負債表失敗: {e}")
        
        # 3. 現金流量表
        try:
            cashflow_url = f"{self.BASE_URL}/mops/web/ajax_t163sb20"
            cashflow_data = {
                'encodeURIComponent': '1',
                'step': '1',
                'firstin': '1',
                'off': '1',
                'co_id': stock_id,
                'year': str(year),
                'season': str(season).zfill(2)
            }
            
            response = requests.post(cashflow_url, data=cashflow_data, headers=self.headers, timeout=10)
            if response.status_code == 200:
                result['cash_flow'] = self._parse_html_table(response.text)
                print(f"✅ 已獲取 {stock_id} {year}Q{season} 現金流量表")
            
        except Exception as e:
            print(f"獲取現金流量表失敗: {e}")
        
        # 儲存快取
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"儲存快取失敗: {e}")
        
        return result
    
    def get_historical_statements(self, stock_id, start_year, end_year, report_type='consolidated'):
        """
        批量獲取多年的歷史財務報表
        
        :param stock_id: 股票代碼
        :param start_year: 開始年度（民國年）
        :param end_year: 結束年度（民國年）
        :param report_type: 報表類型
        :return: list 所有季度的財務報表
        """
        all_statements = []
        
        print(f"\n開始獲取 {stock_id} 從民國 {start_year} 到 {end_year} 的財務報表...")
        print(f"{'='*60}")
        
        for year in range(start_year, end_year + 1):
            for season in range(1, 5):
                print(f"\n📊 正在獲取 {stock_id} - 民國 {year} 年 Q{season}")
                
                try:
                    statement = self.get_financial_statement(stock_id, year, season, report_type)
                    all_statements.append(statement)
                    
                    # 避免請求過快
                    time.sleep(5)
                    
                except Exception as e:
                    print(f"❌ 獲取失敗: {e}")
                    continue
        
        print(f"\n{'='*60}")
        print(f"✨ 完成！共獲取 {len(all_statements)} 個季度的財務報表")
        
        return all_statements
    
    def _parse_html_table(self, html_content):
        """
        解析 HTML 表格為字典
        :param html_content: HTML 內容
        :return: dict 表格數據
        """
        try:
            # 使用 pandas 讀取 HTML 表格
            tables = pd.read_html(io.StringIO(html_content))
            
            if not tables:
                return {}
            
            # 通常第一個表格就是我們要的數據
            df = tables[0]
            
            # 轉換為字典格式
            result = {}
            for idx, row in df.iterrows():
                if len(row) >= 2:
                    key = str(row[0]).strip()
                    value = str(row[1]).strip() if len(row) > 1 else ''
                    
                    # 清理數據（移除逗號、轉換為數字）
                    try:
                        value = value.replace(',', '')
                        value = float(value)
                    except:
                        pass
                    
                    result[key] = value
            
            return result
            
        except Exception as e:
            print(f"解析 HTML 表格失敗: {e}")
            return {}
    
    def get_monthly_revenue(self, stock_id, year, month):
        """
        獲取單月營收
        
        :param stock_id: 股票代碼
        :param year: 民國年
        :param month: 月份 (1-12)
        :return: dict 營收數據
        """
        stock_id = str(stock_id).replace('.TW', '').replace('.TWO', '')
        
        try:
            url = f"{self.BASE_URL}/mops/web/ajax_t05st10"
            data = {
                'encodeURIComponent': '1',
                'step': '1',
                'firstin': '1',
                'off': '1',
                'co_id': stock_id,
                'year': str(year),
                'month': str(month).zfill(2)
            }
            
            response = requests.post(url, data=data, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                tables = pd.read_html(io.StringIO(response.text))
                if tables:
                    df = tables[0]
                    return {
                        'stock_id': stock_id,
                        'year': year,
                        'month': month,
                        'data': df.to_dict('records')
                    }
            
        except Exception as e:
            print(f"獲取月營收失敗: {e}")
        
        return {}
    
    def get_historical_revenue(self, stock_id, start_year, start_month, end_year, end_month):
        """
        批量獲取歷史月營收
        
        :param stock_id: 股票代碼
        :param start_year: 開始年（民國年）
        :param start_month: 開始月
        :param end_year: 結束年（民國年）
        :param end_month: 結束月
        :return: list 所有月份的營收數據
        """
        all_revenue = []
        
        print(f"\n開始獲取 {stock_id} 的歷史月營收...")
        
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            print(f"📅 獲取 {stock_id} - 民國 {current_year} 年 {current_month} 月")
            
            try:
                revenue = self.get_monthly_revenue(stock_id, current_year, current_month)
                if revenue:
                    all_revenue.append(revenue)
                
                time.sleep(3)  # 避免請求過快
                
            except Exception as e:
                print(f"❌ 獲取失敗: {e}")
            
            # 移到下個月
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        print(f"\n✨ 完成！共獲取 {len(all_revenue)} 個月的營收數據")
        
        return all_revenue
    
    @staticmethod
    def ad_to_gregorian(ad_year):
        """民國年轉西元年"""
        return ad_year + 1911
    
    @staticmethod
    def gregorian_to_ad(gregorian_year):
        """西元年轉民國年"""
        return gregorian_year - 1911


# 全局實例
_mops_fetcher = None

def get_mops_fetcher():
    """獲取 MOPS 抓取器單例"""
    global _mops_fetcher
    if _mops_fetcher is None:
        _mops_fetcher = MOPSDataFetcher()
    return _mops_fetcher


# 使用範例
if __name__ == '__main__':
    fetcher = MOPSDataFetcher()
    
    # 範例 1: 獲取台積電 (2330) 2024Q1 的財務報表
    print("範例 1: 獲取單季財務報表")
    result = fetcher.get_financial_statement('2330', 113, 1)  # 113 = 2024
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 範例 2: 獲取台積電 2020-2024 的所有財務報表（5年歷史）
    print("\n範例 2: 獲取多年歷史財務報表")
    historical = fetcher.get_historical_statements('2330', 109, 113)  # 109=2020, 113=2024
    print(f"共獲取 {len(historical)} 季的數據")
    
    # 範例 3: 獲取月營收
    print("\n範例 3: 獲取月營收")
    revenue = fetcher.get_monthly_revenue('2330', 113, 1)
    print(json.dumps(revenue, ensure_ascii=False, indent=2))
