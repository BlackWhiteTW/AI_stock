import os
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta
import requests
from pathlib import Path
from .mops_data import get_mops_fetcher
from .finmind_data import get_finmind_fetcher


class FinancialDataManager:
    """
    財務報表數據管理器（增強版 - 支援多數據源）

    數據源優先級：
    1. Yahoo 財經台灣站（最準確、最即時的台股數據）
    2. yfinance（備用數據源）

    支援功能：
    - 損益表 (Income Statement)
    - 資產負債表 (Balance Sheet)
    - 現金流量表 (Cash Flow Statement)
    - 營收資訊 (Revenue) - 🆕
    - EPS 每股盈餘 - 🆕
    - 財務比率分析
    - 多層快取機制
    """

    def __init__(
        self,
        data_dir="data/financial_data",
        use_yahoo_tw=True,
        use_mops=False,
        use_finmind=True,
    ):
        """
        初始化財務數據管理器
        :param data_dir: 存放財務報表的資料夾路徑（數據直接存於此，不再建立 cache 子資料夾）
        :param use_yahoo_tw: 是否使用 Yahoo 台灣站數據（台股建議開啟）
        :param use_mops: 是否使用證交所 MOPS 數據（實驗性功能，預設關閉）
        :param use_finmind: 是否使用 FinMind API（獲取歷史數據，建議開啟）
        """
        self.data_dir = data_dir
        self.use_yahoo_tw = use_yahoo_tw
        self.use_mops = use_mops
        self.use_finmind = use_finmind

        # 建立目錄（不再建立 cache 子資料夾）
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 🆕 初始化 Yahoo TW 爬蟲
        self.yahoo_scraper = None
        if use_yahoo_tw:
            try:
                from .yahoo_tw_scraper import get_yahoo_scraper

                self.yahoo_scraper = get_yahoo_scraper()
            except Exception as e:
                print(f"Yahoo TW 爬蟲初始化失敗，將使用 yfinance: {e}")
                self.use_yahoo_tw = False

        # 🆕 初始化 MOPS 抓取器（實驗性功能）
        self.mops_fetcher = None
        if use_mops:
            try:
                self.mops_fetcher = get_mops_fetcher()
            except Exception as e:
                print(f"MOPS 抓取器初始化失敗: {e}")
                self.use_mops = False

        # 🆕 初始化 FinMind 抓取器（推薦用於獲取歷史數據）
        self.finmind_fetcher = None
        if use_finmind:
            try:
                self.finmind_fetcher = get_finmind_fetcher()
            except Exception as e:
                print(f"FinMind 抓取器初始化失敗: {e}")
                self.use_finmind = False

    def get_financial_statements(self, ticker, force_update=False):
        """
        獲取股票的完整財務報表（季報）
        數據以 CSV 格式存儲於 data/financial_data/{ticker}/ 目錄下

        :param ticker: 股票代碼（如 '2330.TW'）
        :param force_update: 是否強制更新（忽略快取）
        :return: dict 包含所有財務報表
        """
        # 驗證 ticker 參數
        if not ticker or not isinstance(ticker, str) or ticker.lower() == "unknown":
            print(f"⚠️ 無效的股票代碼: {ticker}")
            return None

        # 建立股票專屬目錄
        stock_dir = os.path.join(self.data_dir, ticker)
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)

        # CSV 檔案路徑
        income_csv = os.path.join(stock_dir, "quarterly_income_stmt.csv")
        balance_csv = os.path.join(stock_dir, "quarterly_balance_sheet.csv")
        cashflow_csv = os.path.join(stock_dir, "quarterly_cashflow.csv")
        info_json = os.path.join(stock_dir, "stock_info.json")

        # 檢查快取（7天內有效）- 優先檢查 info_json（對 ETF 很重要）
        cache_file = info_json if os.path.exists(info_json) else income_csv

        if not force_update and os.path.exists(cache_file):
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(cache_file)
                )
                if cache_age < timedelta(days=7):
                    # 從快取讀取
                    financial_data = {
                        "ticker": ticker,
                        "update_time": datetime.fromtimestamp(
                            os.path.getmtime(cache_file)
                        ).isoformat(),
                        "income_statement": (
                            pd.read_csv(income_csv, encoding="utf-8-sig").to_dict()
                            if os.path.exists(income_csv)
                            else {}
                        ),
                        "balance_sheet": (
                            pd.read_csv(balance_csv, encoding="utf-8-sig").to_dict()
                            if os.path.exists(balance_csv)
                            else {}
                        ),
                        "cash_flow": (
                            pd.read_csv(cashflow_csv, encoding="utf-8-sig").to_dict()
                            if os.path.exists(cashflow_csv)
                            else {}
                        ),
                        "info": {},
                    }

                    # 讀取 info
                    if os.path.exists(info_json):
                        with open(info_json, "r", encoding="utf-8") as f:
                            financial_data["info"] = json.load(f)

                    print(f"✅ 從快取讀取: {ticker}")
                    return financial_data
            except Exception as e:
                print(f"讀取快取失敗: {e}")

        # 從 yfinance 獲取財務報表
        try:
            stock = yf.Ticker(ticker)

            # 獲取數據
            quarterly_income = stock.quarterly_income_stmt
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow

            financial_data = {
                "ticker": ticker,
                "update_time": datetime.now().isoformat(),
                "income_statement": self._df_to_dict(quarterly_income),
                "balance_sheet": self._df_to_dict(quarterly_balance),
                "cash_flow": self._df_to_dict(quarterly_cashflow),
                "annual_income_statement": self._df_to_dict(stock.income_stmt),
                "annual_balance_sheet": self._df_to_dict(stock.balance_sheet),
                "annual_cash_flow": self._df_to_dict(stock.cashflow),
                "info": stock.info,
            }

            # 儲存為 CSV（分別存儲）
            try:
                if quarterly_income is not None and not quarterly_income.empty:
                    quarterly_income.to_csv(income_csv, encoding="utf-8-sig")

                if quarterly_balance is not None and not quarterly_balance.empty:
                    quarterly_balance.to_csv(balance_csv, encoding="utf-8-sig")

                if quarterly_cashflow is not None and not quarterly_cashflow.empty:
                    quarterly_cashflow.to_csv(cashflow_csv, encoding="utf-8-sig")

                # info 存為 JSON
                with open(info_json, "w", encoding="utf-8") as f:
                    json.dump(stock.info, f, ensure_ascii=False, indent=2, default=str)

                print(f"💾 已存至: {stock_dir}/")

            except Exception as e:
                print(f"儲存快取失敗: {e}")

            return financial_data

        except Exception as e:
            print(f"獲取財務報表失敗 [{ticker}]: {e}")
            return None

    def _df_to_dict(self, df):
        """將 DataFrame 轉換為可序列化的字典"""
        if df is None or df.empty:
            return {}

        try:
            # 將 Timestamp 轉為字符串
            df_copy = df.copy()
            df_copy.columns = [str(col) for col in df_copy.columns]
            return df_copy.to_dict()
        except Exception as e:
            return {}

    def _is_etf_ticker(self, ticker):
        """判斷是否為 ETF"""
        try:
            from utils import get_tw_stock_list

            stock_list = get_tw_stock_list()
            if (
                not stock_list.empty
                and "Ticker" in stock_list.columns
                and "產業" in stock_list.columns
            ):
                matched = stock_list[stock_list["Ticker"] == ticker]
                if not matched.empty:
                    industry = str(matched.iloc[0].get("產業", "")).strip().upper()
                    return industry == "ETF"
        except Exception:
            pass

        clean_ticker = ticker.replace(".TW", "").replace(".TWO", "")
        return clean_ticker.startswith("00")

    def _build_etf_summary(self, ticker):
        """建立 ETF 專用摘要，避免傳統財報空白時直接失敗"""
        etf_profile = {
            "source": "Yahoo TW",
            "update_time": datetime.now().isoformat(),
            "holding_count": 0,
            "top_holdings": [],
            "sector_weights": [],
            "raw_summary": {},
            "revenue": {},
            "eps": {},
        }

        if self.yahoo_scraper:
            try:
                etf_data = self.yahoo_scraper.get_all_financial_data(ticker, False)
                if isinstance(etf_data, dict):
                    holding = etf_data.get("holding", {}) or {}
                    etf_profile.update(
                        {
                            "update_time": etf_data.get(
                                "update_time", etf_profile["update_time"]
                            ),
                            "holding_count": len(holding.get("top_holdings", [])),
                            "top_holdings": holding.get("top_holdings", []),
                            "sector_weights": holding.get("sector_weights", []),
                            "raw_summary": holding.get("holdings_summary", {}),
                            "revenue": etf_data.get("revenue", {}),
                            "eps": etf_data.get("eps", {}),
                        }
                    )
            except Exception as e:
                print(f"ETF 摘要獲取失敗 [{ticker}]: {e}")

        return {
            "ticker": ticker,
            "status": "etf",
            "message": "ETF 不適用傳統損益表/資產負債表分析，已改以持股概況呈現。",
            "health_score": 50,
            "rating": "ETF 特別分析",
            "update_time": etf_profile.get("update_time"),
            "latest_period": "N/A",
            "profitability": {"roe": 0, "roa": 0, "net_profit_margin": 0},
            "solvency": {"debt_ratio": 0, "current_ratio": 0},
            "valuation": {"pe_ratio": 0, "pb_ratio": 0, "market_cap": 0},
            "dividend": {"dividend_yield": 0},
            "risk": {"beta": 0},
            "etf_profile": etf_profile,
        }

    def calculate_financial_ratios(self, ticker):
        """
        計算關鍵財務比率

        :param ticker: 股票代碼
        :return: dict 包含各項財務比率
        """
        # 驗證 ticker
        if not ticker or not isinstance(ticker, str) or ticker.lower() == "unknown":
            return {}

        financial_data = self.get_financial_statements(ticker)
        if not financial_data:
            return {}

        ratios = {}

        try:
            # 獲取最新季度的財務數據
            income_stmt = financial_data.get("income_statement", {})
            balance_sheet = financial_data.get("balance_sheet", {})
            cash_flow = financial_data.get("cash_flow", {})

            # 如果數據為空，返回空字典
            if not income_stmt or not balance_sheet:
                return {}

            # 獲取第一個時期（最新）的數據
            latest_period = list(income_stmt.keys())[0] if income_stmt else None

            if not latest_period:
                return {}

            # 提取關鍵數據（使用安全的 get 方法）
            def safe_get(data_dict, period, key, default=0):
                """安全獲取數據"""
                try:
                    period_data = data_dict.get(period, {})
                    value = period_data.get(key, default)
                    return float(value) if value and value != "None" else default
                except:
                    return default

            # 基本財務數據
            total_revenue = safe_get(income_stmt, latest_period, "Total Revenue")
            net_income = safe_get(income_stmt, latest_period, "Net Income")
            total_assets = safe_get(balance_sheet, latest_period, "Total Assets")
            total_liabilities = safe_get(
                balance_sheet, latest_period, "Total Liabilities Net Minority Interest"
            )
            stockholder_equity = safe_get(
                balance_sheet, latest_period, "Stockholders Equity"
            )
            current_assets = safe_get(balance_sheet, latest_period, "Current Assets")
            current_liabilities = safe_get(
                balance_sheet, latest_period, "Current Liabilities"
            )

            # 計算比率
            # 1. 獲利能力指標
            if total_revenue > 0:
                ratios["net_profit_margin"] = (
                    net_income / total_revenue
                ) * 100  # 淨利率 (%)

            if total_assets > 0:
                ratios["roa"] = (net_income / total_assets) * 100  # 資產報酬率 (%)

            if stockholder_equity > 0:
                ratios["roe"] = (
                    net_income / stockholder_equity
                ) * 100  # 股東權益報酬率 (%)

            # 2. 償債能力指標
            if total_assets > 0:
                ratios["debt_ratio"] = (
                    total_liabilities / total_assets
                ) * 100  # 負債比率 (%)

            if current_liabilities > 0:
                ratios["current_ratio"] = (
                    current_assets / current_liabilities
                )  # 流動比率

            # 3. 營運能力指標
            if stockholder_equity > 0 and total_assets > 0:
                ratios["equity_multiplier"] = (
                    total_assets / stockholder_equity
                )  # 權益乘數

            # 4. 公司基本資訊
            info = financial_data.get("info", {})
            ratios["market_cap"] = info.get("marketCap", 0)  # 市值
            ratios["pe_ratio"] = info.get("trailingPE", 0)  # 本益比
            ratios["pb_ratio"] = info.get("priceToBook", 0)  # 股價淨值比
            ratios["dividend_yield"] = info.get("dividendYield", 0)  # 股息殖利率
            ratios["beta"] = info.get("beta", 0)  # Beta 值

            # 添加更新時間
            ratios["update_time"] = datetime.now().isoformat()
            ratios["latest_period"] = latest_period

        except Exception as e:
            print(f"計算財務比率失敗 [{ticker}]: {e}")

        return ratios

    def get_financial_health_score(self, ticker):
        """
        計算財務健康度評分（0-100分）
        綜合考量獲利能力、償債能力、成長性等指標

        :param ticker: 股票代碼
        :return: float 財務健康度評分
        """
        ratios = self.calculate_financial_ratios(ticker)

        if not ratios:
            return 50  # 無數據時返回中性分數

        score = 0

        # 1. 獲利能力 (30分)
        roe = ratios.get("roe", 0)
        if roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe > 5:
            score += 5

        net_margin = ratios.get("net_profit_margin", 0)
        if net_margin > 10:
            score += 15
        elif net_margin > 5:
            score += 10
        elif net_margin > 2:
            score += 5

        # 2. 償債能力 (25分)
        debt_ratio = ratios.get("debt_ratio", 100)
        if debt_ratio < 30:
            score += 15
        elif debt_ratio < 50:
            score += 10
        elif debt_ratio < 70:
            score += 5

        current_ratio = ratios.get("current_ratio", 0)
        if current_ratio > 2:
            score += 10
        elif current_ratio > 1.5:
            score += 7
        elif current_ratio > 1:
            score += 3

        # 3. 估值吸引力 (25分)
        pe_ratio = ratios.get("pe_ratio", 0)
        if 0 < pe_ratio < 15:
            score += 15
        elif 15 <= pe_ratio < 25:
            score += 10
        elif 25 <= pe_ratio < 35:
            score += 5

        pb_ratio = ratios.get("pb_ratio", 0)
        if 0 < pb_ratio < 1.5:
            score += 10
        elif 1.5 <= pb_ratio < 3:
            score += 7
        elif 3 <= pb_ratio < 5:
            score += 3

        # 4. 股息吸引力 (20分)
        dividend_yield = ratios.get("dividend_yield", 0) * 100  # 轉為百分比
        if dividend_yield > 5:
            score += 20
        elif dividend_yield > 3:
            score += 15
        elif dividend_yield > 2:
            score += 10
        elif dividend_yield > 1:
            score += 5

        return min(score, 100)  # 確保不超過100分

    def get_financial_summary(self, ticker):
        """
        獲取財務數據摘要（用於顯示）

        :param ticker: 股票代碼
        :return: dict 財務摘要
        """
        if self._is_etf_ticker(ticker):
            return self._build_etf_summary(ticker)

        ratios = self.calculate_financial_ratios(ticker)
        health_score = self.get_financial_health_score(ticker)

        if not ratios:
            return {
                "ticker": ticker,
                "status": "no_data",
                "message": "無法獲取財務數據",
            }

        # 評級系統
        def get_rating(score):
            if score >= 80:
                return "A+ 優秀"
            elif score >= 70:
                return "A 良好"
            elif score >= 60:
                return "B+ 中上"
            elif score >= 50:
                return "B 中等"
            elif score >= 40:
                return "C 中下"
            else:
                return "D 不佳"

        summary = {
            "ticker": ticker,
            "health_score": round(health_score, 1),
            "rating": get_rating(health_score),
            "update_time": ratios.get("update_time"),
            "latest_period": ratios.get("latest_period"),
            # 獲利能力
            "profitability": {
                "roe": round(ratios.get("roe", 0), 2),
                "roa": round(ratios.get("roa", 0), 2),
                "net_profit_margin": round(ratios.get("net_profit_margin", 0), 2),
            },
            # 償債能力
            "solvency": {
                "debt_ratio": round(ratios.get("debt_ratio", 0), 2),
                "current_ratio": round(ratios.get("current_ratio", 0), 2),
            },
            # 估值指標
            "valuation": {
                "pe_ratio": round(ratios.get("pe_ratio", 0), 2),
                "pb_ratio": round(ratios.get("pb_ratio", 0), 2),
                "market_cap": ratios.get("market_cap", 0),
            },
            # 股息資訊
            "dividend": {
                "dividend_yield": round(ratios.get("dividend_yield", 0) * 100, 2)
            },
            # 風險指標
            "risk": {"beta": round(ratios.get("beta", 0), 2)},
        }

        return summary

    def get_financial_features(self, ticker):
        """
        提取用於 AI 模型的財務特徵

        :param ticker: 股票代碼
        :return: dict 標準化的財務特徵
        """
        ratios = self.calculate_financial_ratios(ticker)

        if not ratios:
            # 返回中性值
            return {
                "financial_health_score": 50,
                "roe_normalized": 0,
                "debt_ratio_normalized": 0,
                "current_ratio_normalized": 0,
                "pe_ratio_normalized": 0,
                "dividend_yield_normalized": 0,
            }

        # 標準化特徵（0-100 或 0-1）
        features = {
            "financial_health_score": self.get_financial_health_score(ticker),
            # ROE 標準化 (假設最佳值為 20%)
            "roe_normalized": min(max(ratios.get("roe", 0) / 20 * 100, 0), 100),
            # 負債比率標準化 (越低越好，反轉)
            "debt_ratio_normalized": max(100 - ratios.get("debt_ratio", 50), 0),
            # 流動比率標準化 (假設最佳值為 2)
            "current_ratio_normalized": min(
                ratios.get("current_ratio", 1) / 2 * 100, 100
            ),
            # 本益比標準化 (假設合理值為 15)
            "pe_ratio_normalized": (
                min(15 / max(ratios.get("pe_ratio", 15), 1) * 100, 100)
                if ratios.get("pe_ratio", 0) > 0
                else 50
            ),
            # 股息殖利率標準化 (假設優秀值為 5%)
            "dividend_yield_normalized": min(
                ratios.get("dividend_yield", 0) * 100 / 5 * 100, 100
            ),
        }

        return features

    def get_revenue_data(self, ticker, force_update=False):
        """
        🆕 獲取營收資訊（優先使用 Yahoo TW 數據）

        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含營收數據
        """
        # 優先使用 Yahoo TW 數據
        if self.use_yahoo_tw and self.yahoo_scraper:
            try:
                revenue_data = self.yahoo_scraper.get_revenue(ticker, force_update)
                if revenue_data and revenue_data.get("monthly_revenue"):
                    return revenue_data
            except Exception as e:
                print(f"Yahoo TW 營收數據獲取失敗，嘗試其他方式: {e}")

        # 備用：從 yfinance 獲取（但可能不完整）
        try:
            financial_data = self.get_financial_statements(ticker, force_update)
            if financial_data:
                income_stmt = financial_data.get("income_statement", {})
                return {
                    "ticker": ticker,
                    "source": "yfinance",
                    "update_time": datetime.now().isoformat(),
                    "quarterly_revenue": income_stmt,
                }
        except Exception as e:
            print(f"獲取營收數據失敗: {e}")

        return {}

    def get_eps_data(self, ticker, force_update=False):
        """
        🆕 獲取 EPS（每股盈餘）資訊（優先使用 Yahoo TW 數據）

        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 包含 EPS 數據
        """
        # 優先使用 Yahoo TW 數據
        if self.use_yahoo_tw and self.yahoo_scraper:
            try:
                eps_data = self.yahoo_scraper.get_eps(ticker, force_update)
                if eps_data and eps_data.get("quarterly_eps"):
                    return eps_data
            except Exception as e:
                print(f"Yahoo TW EPS 數據獲取失敗，嘗試其他方式: {e}")

        # 備用：從基本資訊獲取
        try:
            financial_data = self.get_financial_statements(ticker, force_update)
            if financial_data:
                info = financial_data.get("info", {})
                return {
                    "ticker": ticker,
                    "source": "yfinance",
                    "update_time": datetime.now().isoformat(),
                    "trailing_eps": info.get("trailingEps", 0),
                    "forward_eps": info.get("forwardEps", 0),
                }
        except Exception as e:
            print(f"獲取 EPS 數據失敗: {e}")

        return {}

    def get_comprehensive_financial_data(self, ticker, force_update=False):
        """
        🆕 獲取綜合財務數據（整合所有數據源）

        包含：
        - 基本財務報表（yfinance）
        - Yahoo TW 營收數據
        - Yahoo TW EPS 數據
        - Yahoo TW 三大報表（更準確）
        - 財務比率與評分

        :param ticker: 股票代碼
        :param force_update: 是否強制更新
        :return: dict 綜合財務數據
        """
        comprehensive_data = {
            "ticker": ticker,
            "update_time": datetime.now().isoformat(),
            "data_sources": [],
        }

        # 1. Yahoo TW 數據（台股專用）
        if self.use_yahoo_tw and self.yahoo_scraper:
            try:
                yahoo_data = self.yahoo_scraper.get_all_financial_data(
                    ticker, force_update
                )
                comprehensive_data["yahoo_tw"] = yahoo_data
                comprehensive_data["data_sources"].append("Yahoo TW")
            except Exception as e:
                print(f"Yahoo TW 綜合數據獲取失敗: {e}")

        # 2. yfinance 數據（國際數據）
        try:
            yf_data = self.get_financial_statements(ticker, force_update)
            if yf_data:
                comprehensive_data["yfinance"] = yf_data
                comprehensive_data["data_sources"].append("yfinance")
        except Exception as e:
            print(f"yfinance 數據獲取失敗: {e}")

        # 3. 財務分析
        try:
            comprehensive_data["analysis"] = {
                "ratios": self.calculate_financial_ratios(ticker),
                "health_score": self.get_financial_health_score(ticker),
                "summary": self.get_financial_summary(ticker),
            }
        except Exception as e:
            print(f"財務分析失敗: {e}")

        return comprehensive_data

    def batch_update_financial_data(self, ticker_list):
        """
        批量更新多支股票的財務數據

        :param ticker_list: 股票代碼列表
        :return: dict 更新結果統計
        """
        results = {"success": [], "failed": [], "total": len(ticker_list)}

        for ticker in ticker_list:
            try:
                data = self.get_financial_statements(ticker, force_update=True)
                if data:
                    results["success"].append(ticker)
                else:
                    results["failed"].append(ticker)
            except Exception as e:
                print(f"更新 {ticker} 失敗: {e}")
                results["failed"].append(ticker)

        return results

    def get_historical_financial_data(
        self, ticker, start_year, end_year=None, use_finmind=True
    ):
        """
        獲取指定股票的歷史財務報表

        優先使用 FinMind API（快速、穩定、完整），
        如果失敗則嘗試使用 MOPS（較慢但完整）

        :param ticker: 股票代碼（如 '2330.TW'）
        :param start_year: 開始年份（西元年，如 2020）
        :param end_year: 結束年份（西元年，如 2024），不指定則到當年
        :param use_finmind: 是否優先使用 FinMind（推薦）
        :return: dict 包含多年的歷史財務報表
        """
        # 清理股票代碼
        stock_id = ticker.replace(".TW", "").replace(".TWO", "")

        if end_year is None:
            end_year = datetime.now().year

        print(f"\n{'='*60}")
        print(f"📊 開始獲取 {ticker} 的歷史財務數據")
        print(f"時間範圍: {start_year} - {end_year}")
        print(f"{'='*60}\n")

        # 方案 1: 優先使用 FinMind API（推薦）
        if use_finmind and self.use_finmind and self.finmind_fetcher:
            try:
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"

                result = self.finmind_fetcher.get_all_historical_data(
                    stock_id, start_date, end_date
                )

                if result:
                    # 轉換為標準格式
                    financial_data = {
                        "ticker": ticker,
                        "stock_id": stock_id,
                        "start_year": start_year,
                        "end_year": end_year,
                        "data_source": "FinMind",
                        "fetch_time": datetime.now().isoformat(),
                        "financial_statements": result.get("financial_statements"),
                        "balance_sheet": result.get("balance_sheet"),
                        "cash_flow": result.get("cash_flow"),
                        "monthly_revenue": result.get("monthly_revenue"),
                    }

                    # CSV 檔案已經在 finmind_data.py 中分別存儲
                    # 不需要再額外保存

                    print(f"\n✅ 成功使用 FinMind 獲取歷史數據")
                    print(f"📂 數據已分別存儲至: data/financial_data/{ticker}/")
                    return financial_data

            except Exception as e:
                print(f"⚠️ FinMind 獲取失敗: {e}")

        # 方案 2: 備用方案 - 使用 MOPS（較慢）
        if self.use_mops and self.mops_fetcher:
            try:
                # 轉換為民國年
                start_ad = self.mops_fetcher.gregorian_to_ad(start_year)
                end_ad = self.mops_fetcher.gregorian_to_ad(end_year)

                print(f"\n⚠️ 使用備用方案: MOPS 數據源（此過程較慢，請耐心等待...）")

                historical_statements = self.mops_fetcher.get_historical_statements(
                    stock_id, start_ad, end_ad
                )

                result = {
                    "ticker": ticker,
                    "stock_id": stock_id,
                    "start_year": start_year,
                    "end_year": end_year,
                    "data_source": "MOPS",
                    "total_periods": len(historical_statements),
                    "statements": historical_statements,
                    "fetch_time": datetime.now().isoformat(),
                }

                # 儲存快取
                cache_file = os.path.join(
                    self.cache_dir,
                    f"{ticker}_historical_{start_year}_{end_year}_mops.json",
                )

                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    print(f"\n💾 已儲存歷史數據至: {cache_file}")
                except Exception as e:
                    print(f"儲存歷史數據失敗: {e}")

                return result

            except Exception as e:
                print(f"❌ MOPS 獲取失敗: {e}")

        print(f"\n❌ 所有數據源都無法獲取歷史財務數據")
        return None

    def get_historical_revenue(
        self, ticker, start_year, start_month=1, end_year=None, end_month=12
    ):
        """
        獲取歷史月營收數據（使用 MOPS 數據）

        :param ticker: 股票代碼
        :param start_year: 開始年份（西元年）
        :param start_month: 開始月份
        :param end_year: 結束年份（西元年），不指定則到當年
        :param end_month: 結束月份
        :return: dict 包含歷史月營收數據
        """
        if not self.use_mops or not self.mops_fetcher:
            print("⚠️ MOPS 數據源未啟用")
            return None

        stock_id = ticker.replace(".TW", "").replace(".TWO", "")

        # 轉換為民國年
        start_ad = self.mops_fetcher.gregorian_to_ad(start_year)

        if end_year is None:
            end_year = datetime.now().year
        end_ad = self.mops_fetcher.gregorian_to_ad(end_year)

        try:
            revenue_data = self.mops_fetcher.get_historical_revenue(
                stock_id, start_ad, start_month, end_ad, end_month
            )

            result = {
                "ticker": ticker,
                "stock_id": stock_id,
                "start_period": f"{start_year}/{start_month}",
                "end_period": f"{end_year}/{end_month}",
                "total_months": len(revenue_data),
                "data": revenue_data,
                "fetch_time": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            print(f"獲取歷史營收數據失敗: {e}")
            return None


# 全局實例（單例模式）
_financial_manager = None


def get_financial_manager():
    """獲取財務數據管理器單例"""
    global _financial_manager
    if _financial_manager is None:
        _financial_manager = FinancialDataManager()
    return _financial_manager
