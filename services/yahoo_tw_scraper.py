"""
Yahoo 財經台灣站數據爬蟲
專門用於抓取台股的財務報表、營收、EPS 等官方資訊
"""

import requests
import pandas as pd
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
import re


class YahooTWFinancialScraper:
    """
    Yahoo 財經台灣站財務數據爬蟲
    支援多種財務報表的抓取與快取
    """

    BASE_URL = "https://tw.stock.yahoo.com"

    def __init__(self, cache_dir="data/yahoo_tw_cache"):
        """
        初始化爬蟲
        :param cache_dir: 快取目錄
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 設定 User-Agent 避免被擋
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
        }

        # 快取有效期設定（天數）
        self.cache_expiry = {
            "revenue": 30,  # 營收：每月更新
            "eps": 90,  # EPS：每季更新
            "income": 90,  # 損益表：每季更新
            "balance": 90,  # 資產負債表：每季更新
            "cashflow": 90,  # 現金流量表：每季更新
            "holding": 7,  # 持股明細：每週更新（ETF 成分股變化）
        }

    def _get_cache_path(self, ticker, data_type):
        """獲取快取檔案路徑"""
        # 標準化 ticker（移除 .TW 後綴）
        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        return self.cache_dir / f"{clean_ticker}_{data_type}.json"

    def _is_cache_valid(self, cache_path, data_type):
        """檢查快取是否有效"""
        if not cache_path.exists():
            return False

        try:
            # 檢查檔案修改時間
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            cache_age = datetime.now() - mtime
            expiry_days = self.cache_expiry.get(data_type, 7)

            return cache_age < timedelta(days=expiry_days)
        except Exception as e:
            print(f"檢查快取失敗: {e}")
            return False

    def _load_cache(self, cache_path):
        """載入快取數據"""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"載入快取失敗: {e}")
            return None

    def _save_cache(self, cache_path, data):
        """儲存快取數據"""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception as e:
            print(f"儲存快取失敗: {e}")
            return False

    def _fetch_page(self, url, max_retries=3):
        """
        抓取網頁內容
        :param url: 網址
        :param max_retries: 最大重試次數
        :return: requests.Response 或 None
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response
                else:
                    print(f"HTTP {response.status_code}: {url}")
            except requests.RequestException as e:
                print(f"請求失敗 (嘗試 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # 指數退避

        return None

    def get_revenue(self, ticker, force_update=False):
        """
        獲取營收資訊
        :param ticker: 股票代碼（如 '2330.TW' 或 '2330'）
        :param force_update: 是否強制更新
        :return: dict 包含營收數據
        """
        cache_path = self._get_cache_path(ticker, "revenue")

        # 檢查快取
        if not force_update and self._is_cache_valid(cache_path, "revenue"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        # 標準化 ticker
        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/revenue"

        print(f"正在抓取營收數據: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            # 解析 HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 尋找營收表格
            revenue_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "monthly_revenue": [],
                "yearly_revenue": [],
            }

            # 查找表格（Yahoo 財經使用特定的 class）
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                # 解析表頭
                headers = [
                    th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                ]

                # 解析數據行
                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols) >= len(headers):
                        row_data = dict(zip(headers, cols))
                        revenue_data["monthly_revenue"].append(row_data)

            # 儲存快取
            self._save_cache(cache_path, revenue_data)
            return revenue_data

        except Exception as e:
            print(f"解析營收數據失敗: {e}")
            return self._load_cache(cache_path) or {}

    def get_eps(self, ticker, force_update=False):
        """
        獲取 EPS（每股盈餘）資訊
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含 EPS 數據
        """
        cache_path = self._get_cache_path(ticker, "eps")

        # 檢查快取
        if not force_update and self._is_cache_valid(cache_path, "eps"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/eps"

        print(f"正在抓取 EPS 數據: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            eps_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "quarterly_eps": [],
                "yearly_eps": [],
            }

            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                headers = [
                    th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                ]

                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols) >= len(headers):
                        row_data = dict(zip(headers, cols))
                        eps_data["quarterly_eps"].append(row_data)

            self._save_cache(cache_path, eps_data)
            return eps_data

        except Exception as e:
            print(f"解析 EPS 數據失敗: {e}")
            return self._load_cache(cache_path) or {}

    def get_income_statement(self, ticker, force_update=False):
        """
        獲取損益表
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含損益表數據
        """
        cache_path = self._get_cache_path(ticker, "income")

        if not force_update and self._is_cache_valid(cache_path, "income"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/income-statement"

        print(f"正在抓取損益表: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            statement_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "periods": [],
                "data": [],
            }

            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                # 第一行是期間（季度/年度）
                headers = [
                    th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                ]
                if headers:
                    statement_data["periods"] = headers

                # 後續行是各項目數據
                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        statement_data["data"].append(cols)

            self._save_cache(cache_path, statement_data)
            return statement_data

        except Exception as e:
            print(f"解析損益表失敗: {e}")
            return self._load_cache(cache_path) or {}

    def get_balance_sheet(self, ticker, force_update=False):
        """
        獲取資產負債表
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含資產負債表數據
        """
        cache_path = self._get_cache_path(ticker, "balance")

        if not force_update and self._is_cache_valid(cache_path, "balance"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/balance-sheet"

        print(f"正在抓取資產負債表: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            balance_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "periods": [],
                "data": [],
            }

            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                headers = [
                    th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                ]
                if headers:
                    balance_data["periods"] = headers

                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        balance_data["data"].append(cols)

            self._save_cache(cache_path, balance_data)
            return balance_data

        except Exception as e:
            print(f"解析資產負債表失敗: {e}")
            return self._load_cache(cache_path) or {}

    def get_cash_flow_statement(self, ticker, force_update=False):
        """
        獲取現金流量表
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含現金流量表數據
        """
        cache_path = self._get_cache_path(ticker, "cashflow")

        if not force_update and self._is_cache_valid(cache_path, "cashflow"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/cash-flow-statement"

        print(f"正在抓取現金流量表: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            cashflow_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "periods": [],
                "data": [],
            }

            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                if not rows:
                    continue

                headers = [
                    th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                ]
                if headers:
                    cashflow_data["periods"] = headers

                for row in rows[1:]:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if cols:
                        cashflow_data["data"].append(cols)

            self._save_cache(cache_path, cashflow_data)
            return cashflow_data

        except Exception as e:
            print(f"解析現金流量表失敗: {e}")
            return self._load_cache(cache_path) or {}

    def get_holding(self, ticker, force_update=False):
        """
        獲取持股明細（主要用於 ETF）
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含持股明細數據
        """
        cache_path = self._get_cache_path(ticker, "holding")

        if not force_update and self._is_cache_valid(cache_path, "holding"):
            cached_data = self._load_cache(cache_path)
            if cached_data:
                return cached_data

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        url = f"{self.BASE_URL}/quote/{clean_ticker}.TW/holding"

        print(f"正在抓取持股明細: {url}")
        response = self._fetch_page(url)

        if not response:
            return self._load_cache(cache_path) or {}

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            holding_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "top_holdings": [],  # 前十大持股
                "sector_weights": [],  # 產業配置
                "holdings_summary": {},  # 持股摘要
            }

            # wantgoo 會將持股渲染到 holdingTable，先優先解析這個區塊
            holding_rows = soup.select("#holdingTable tr")
            if holding_rows:
                parsed_rows = self._parse_wantgoo_holding_rows(holding_rows)
                if parsed_rows:
                    holding_data["top_holdings"] = parsed_rows

            # 若 holdingTable 尚未注入，再回退掃描一般表格
            if not holding_data["top_holdings"]:
                # 查找所有表格
                tables = soup.find_all("table")

                for table in tables:
                    rows = table.find_all("tr")
                    if not rows:
                        continue

                    # 解析表頭
                    headers = [
                        th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])
                    ]

                    # 判斷表格類型（根據表頭內容）
                    if headers and any(
                        keyword in "".join(headers)
                        for keyword in ["股票", "代號", "比重", "權重"]
                    ):
                        # 這是持股明細表
                        for row in rows[1:]:
                            cols = [
                                td.get_text(strip=True) for td in row.find_all("td")
                            ]
                            if len(cols) >= 2:
                                row_data = dict(zip(headers[: len(cols)], cols))
                                holding_data["top_holdings"].append(row_data)

                    elif headers and any(
                        keyword in "".join(headers)
                        for keyword in ["產業", "類別", "分類"]
                    ):
                        # 這是產業配置表
                        for row in rows[1:]:
                            cols = [
                                td.get_text(strip=True) for td in row.find_all("td")
                            ]
                            if len(cols) >= 2:
                                row_data = dict(zip(headers[: len(cols)], cols))
                                holding_data["sector_weights"].append(row_data)

            # 查找持股摘要資訊（如總持股數、總市值等）
            # 查找包含摘要資訊的 div 或其他容器
            summary_divs = soup.find_all(
                ["div", "span"], class_=re.compile("summary|info|detail", re.I)
            )
            for div in summary_divs:
                text = div.get_text(strip=True)
                # 提取可能的摘要資訊
                if "總市值" in text or "持股數" in text or "檔數" in text:
                    holding_data["holdings_summary"]["raw_text"] = text

            self._save_cache(cache_path, holding_data)
            return holding_data

        except Exception as e:
            print(f"解析持股明細失敗: {e}")
            return self._load_cache(cache_path) or {}

    def _parse_wantgoo_holding_rows(self, rows):
        """解析 wantgoo holdingTable 的單列持股內容。"""
        parsed_rows = []

        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            texts = [cell.get_text(" ", strip=True) for cell in cells]
            if not any(texts):
                continue

            # 跳過表頭列
            first_text = texts[0]
            if "代號" in first_text and "名稱" in first_text:
                continue

            links = row.find_all("a", href=True)
            row_url = links[0]["href"] if links else None

            stock_no = texts[0] if len(texts) > 0 else ""
            name = texts[1] if len(texts) > 1 else ""
            shareholding_rate = texts[2] if len(texts) > 2 else ""
            holding_change = texts[3] if len(texts) > 3 else ""
            close_price = texts[4] if len(texts) > 4 else ""
            change_percent_1 = texts[5] if len(texts) > 5 else ""
            change_percent_20 = texts[6] if len(texts) > 6 else ""
            change_percent_60 = texts[7] if len(texts) > 7 else ""

            parsed_rows.append(
                {
                    "code": stock_no,
                    "name": name,
                    "percentage": shareholding_rate,
                    "shareholdingRate": shareholding_rate,
                    "holdingChange": holding_change,
                    "close": close_price,
                    "changePercent1": change_percent_1,
                    "changePercent20": change_percent_20,
                    "changePercent60": change_percent_60,
                    "stockNo": stock_no,
                    "url": row_url,
                    "raw": texts,
                }
            )

        return parsed_rows

    def get_all_financial_data(self, ticker, force_update=False):
        """
        一次獲取所有財務數據
        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含所有財務數據
        """
        all_data = {
            "ticker": ticker,
            "update_time": datetime.now().isoformat(),
            "revenue": self.get_revenue(ticker, force_update),
            "eps": self.get_eps(ticker, force_update),
            "income_statement": self.get_income_statement(ticker, force_update),
            "balance_sheet": self.get_balance_sheet(ticker, force_update),
            "cash_flow": self.get_cash_flow_statement(ticker, force_update),
            "holding": self.get_holding(ticker, force_update),  # 🆕 持股明細
        }

        return all_data

    def parse_revenue_growth(self, revenue_data):
        """
        分析營收成長率
        :param revenue_data: 營收數據
        :return: dict 包含成長率分析
        """
        try:
            monthly_revenue = revenue_data.get("monthly_revenue", [])
            if not monthly_revenue:
                return {}

            # 提取最近的營收數據進行分析
            analysis = {
                "latest_month": None,
                "mom_growth": None,  # 月增率
                "yoy_growth": None,  # 年增率
                "avg_3m_growth": None,  # 近3月平均年增率
            }

            # 這裡需要根據實際的 HTML 結構來解析
            # 暫時返回空字典，等實際測試後再完善

            return analysis

        except Exception as e:
            print(f"分析營收成長率失敗: {e}")
            return {}

    def clear_expired_cache(self):
        """清理過期的快取檔案"""
        cleared_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                # 從檔名提取數據類型
                filename = cache_file.stem
                parts = filename.split("_")
                if len(parts) >= 2:
                    data_type = parts[-1]

                    if not self._is_cache_valid(cache_file, data_type):
                        cache_file.unlink()
                        cleared_count += 1
                        print(f"已清理過期快取: {cache_file.name}")
            except Exception as e:
                print(f"清理快取失敗 {cache_file}: {e}")

        return cleared_count


# 全局實例（單例模式）
_yahoo_scraper = None


def get_yahoo_scraper():
    """獲取 Yahoo 財經爬蟲單例"""
    global _yahoo_scraper
    if _yahoo_scraper is None:
        _yahoo_scraper = YahooTWFinancialScraper()
    return _yahoo_scraper
