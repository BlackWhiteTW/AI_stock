"""
📰 周期性新聞收集器 (Weekly News Collector) v1.0
自動收集 Google 新聞並按周存儲，供 AI 模型訓練使用
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from services.search_engine import SearchEngine
from services.news_analyzer import NewsAnalyzer


class WeeklyNewsCollector:
    """
    周期性新聞收集器
    - 每天自動收集新聞
    - 按 ISO 周（例如 2026_W10）組織檔案
    - 一周後自動切換到新檔案
    - 提供新聞情緒分數供模型訓練使用
    """

    def __init__(self, base_dir="data/weekly_news"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.search_engine = SearchEngine()
        self.news_analyzer = NewsAnalyzer()

        # 預設搜尋關鍵詞類別
        self.search_categories = {
            "台股市場": ["台股", "台灣股市", "加權指數", "台股行情"],
            "科技產業": ["台積電", "半導體", "IC設計", "AI晶片"],
            "金融產業": ["金融股", "銀行股", "保險股", "金控"],
            "傳產製造": ["塑化", "鋼鐵", "水泥", "航運"],
            "電子零組件": ["PCB", "被動元件", "電源供應", "連接器"],
            "國際市場": ["美股", "那斯達克", "道瓊", "S&P500", "聯準會"],
            "總體經濟": ["通膨", "升息", "降息", "GDP", "失業率"],
        }

    def get_current_week_filename(self) -> str:
        """
        獲取當前周的檔案名稱
        格式: YYYY_WXX.json (例如: 2026_W10.json)
        """
        now = datetime.now()
        year, week, _ = now.isocalendar()
        return f"{year}_W{week:02d}.json"

    def get_week_file_path(self, week_filename: Optional[str] = None) -> Path:
        """獲取周檔案的完整路徑"""
        if week_filename is None:
            week_filename = self.get_current_week_filename()
        return self.base_dir / week_filename

    def load_week_data(self, week_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        載入周新聞數據

        返回格式:
        {
            "week": "2026_W10",
            "start_date": "2026-03-02",
            "end_date": "2026-03-08",
            "news_items": [],
            "last_updated": "2026-03-07 10:30:00"
        }
        """
        file_path = self.get_week_file_path(week_filename)

        if not file_path.exists():
            # 創建新的周檔案
            year, week = self._parse_week_filename(
                week_filename or self.get_current_week_filename()
            )
            start_date, end_date = self._get_week_date_range(year, week)

            return {
                "week": week_filename or self.get_current_week_filename(),
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "news_items": [],
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "collection_count": 0,
            }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 載入周數據失敗: {e}")
            return self.load_week_data(None)  # 返回空數據結構

    def save_week_data(self, data: Dict[str, Any], week_filename: Optional[str] = None):
        """保存周新聞數據"""
        file_path = self.get_week_file_path(week_filename)
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 周新聞數據已保存: {file_path}")
        except Exception as e:
            print(f"❌ 保存周數據失敗: {e}")

    def collect_daily_news(
        self, custom_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        每日新聞收集任務

        參數:
            custom_queries: 自定義搜尋關鍵詞列表（可選）

        返回:
            收集統計資訊
        """
        print("=" * 80)
        print(f"📰 開始每日新聞收集 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 載入當前周數據
        week_data = self.load_week_data()

        # 準備搜尋關鍵詞
        queries = custom_queries if custom_queries else self._generate_search_queries()

        news_collected = 0
        news_items = week_data.get("news_items", [])
        existing_hashes = {item.get("content_hash") for item in news_items}

        # ⚡ 快速失敗機制：連續失敗超過 3 次就跳過整個收集
        consecutive_failures = 0
        max_consecutive_failures = 3

        for category, keywords in self.search_categories.items():
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⏭️ 連續失敗 {max_consecutive_failures} 次，跳過剩餘類別")
                break

            print(f"\n🔍 搜尋類別: {category}")

            for keyword in keywords[:2]:  # 每個類別取前2個關鍵詞
                try:
                    # 使用 SearchEngine 獲取新聞
                    news_content = self.search_engine.get_latest_news(
                        query=f"{keyword} 台灣", num_results=5
                    )

                    if not news_content:
                        consecutive_failures += 1
                        print(
                            f"  ⏭️ {keyword}: API 無回應，跳過 (失敗次數: {consecutive_failures})"
                        )
                        continue

                    # 重置失敗計數器（成功獲取數據）
                    consecutive_failures = 0

                    # 分析新聞情緒
                    sentiment_score = self.news_analyzer.analyze_text_sentiment(
                        news_content
                    )

                    # 創建新聞項目
                    content_hash = hashlib.md5(news_content.encode()).hexdigest()

                    # 避免重複
                    if content_hash in existing_hashes:
                        continue

                    news_item = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "category": category,
                        "keyword": keyword,
                        "content": news_content,
                        "content_hash": content_hash,
                        "sentiment_score": sentiment_score,
                        "source": "Google News (SerpAPI)",
                    }

                    news_items.append(news_item)
                    existing_hashes.add(content_hash)
                    news_collected += 1

                    print(f"  ✅ {keyword}: 情緒分數 {sentiment_score:.2f}")

                except Exception as e:
                    consecutive_failures += 1
                    print(
                        f"  ⚠️ {keyword} 搜尋失敗: {str(e)[:50]} (失敗次數: {consecutive_failures})"
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"  🛑 達到最大失敗次數，停止收集")
                        break

        # 更新周數據
        week_data["news_items"] = news_items
        week_data["collection_count"] = week_data.get("collection_count", 0) + 1
        self.save_week_data(week_data)

        stats = {
            "week": week_data["week"],
            "news_collected_today": news_collected,
            "total_news_this_week": len(news_items),
            "last_updated": week_data["last_updated"],
        }

        print("\n" + "=" * 80)
        print(f"📊 收集統計:")
        print(f"   本次收集: {news_collected} 則")
        print(f"   本周累計: {len(news_items)} 則")
        print(f"   周檔案: {week_data['week']}")
        print("=" * 80)

        return stats

    def collect_stock_specific_news(
        self, ticker: str, stock_name: str
    ) -> Dict[str, Any]:
        """
        收集特定股票的新聞

        參數:
            ticker: 股票代碼（例如：2330.TW）
            stock_name: 股票名稱（例如：台積電）

        返回:
            包含新聞內容和情緒分數的字典
        """
        ticker_clean = ticker.replace(".TW", "").replace(".TWO", "")

        queries = [
            f"{stock_name}",
            f"{ticker_clean} 股價",
            f"{stock_name} 營收",
            f"{stock_name} 展望",
        ]

        all_news = []
        sentiment_scores = []

        # ⚡ 添加超時保護
        for query in queries:
            try:
                news_content = self.search_engine.get_latest_news(query, num_results=3)

                if news_content:  # None 或空字串都會被跳過
                    sentiment_score = self.news_analyzer.analyze_text_sentiment(
                        news_content
                    )
                    all_news.append(news_content)
                    sentiment_scores.append(sentiment_score)
                else:
                    print(f"  ⏭️ {query}: 無數據，跳過")
            except Exception as e:
                print(f"⚠️ {query} 搜尋失敗: {str(e)[:50]}")
                continue  # 繼續下一個查詢，不中斷

        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        )

        return {
            "ticker": ticker,
            "stock_name": stock_name,
            "news_items": all_news,
            "sentiment_score": avg_sentiment,
            "news_count": len(all_news),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_recent_news_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        獲取最近幾天的新聞摘要

        參數:
            days: 回顧天數（預設 7 天）

        返回:
            新聞摘要統計
        """
        # 載入當前周
        week_data = self.load_week_data()

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_news = [
            item
            for item in week_data.get("news_items", [])
            if datetime.strptime(item["date"], "%Y-%m-%d") >= cutoff_date
        ]

        # 計算平均情緒
        if recent_news:
            avg_sentiment = sum(item["sentiment_score"] for item in recent_news) / len(
                recent_news
            )

            # 按類別分組
            by_category = {}
            for item in recent_news:
                category = item.get("category", "其他")
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(item)
        else:
            avg_sentiment = 0.0
            by_category = {}

        return {
            "period": f"最近 {days} 天",
            "total_news": len(recent_news),
            "average_sentiment": avg_sentiment,
            "by_category": {
                cat: {
                    "count": len(items),
                    "avg_sentiment": sum(i["sentiment_score"] for i in items)
                    / len(items),
                }
                for cat, items in by_category.items()
            },
            "latest_update": week_data.get("last_updated", "N/A"),
        }

    def get_sentiment_features_for_training(
        self, lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        為模型訓練提供新聞情緒特徵

        參數:
            lookback_days: 回顧天數

        返回:
            特徵字典，可直接用於模型訓練
        """
        summary = self.get_recent_news_summary(days=lookback_days)

        features = {
            "news_sentiment_7d": summary["average_sentiment"],
            "news_count_7d": summary["total_news"],
            "news_intensity": min(summary["total_news"] / 50.0, 1.0),  # 標準化到 0-1
        }

        # 添加各類別情緒
        for category, stats in summary["by_category"].items():
            feature_name = f"news_sentiment_{category.replace(' ', '_')}"
            features[feature_name] = stats["avg_sentiment"]

        return features

    def cleanup_old_files(self, keep_weeks: int = 12):
        """
        清理舊的周檔案（保留最近 N 周）

        參數:
            keep_weeks: 保留最近幾周的檔案（預設 12 周，約 3 個月）
        """
        if not self.base_dir.exists():
            return

        all_files = sorted(self.base_dir.glob("*.json"), reverse=True)

        if len(all_files) <= keep_weeks:
            print(
                f"✅ 周檔案數量 ({len(all_files)}) 未超過限制 ({keep_weeks})，無需清理"
            )
            return

        files_to_delete = all_files[keep_weeks:]

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"🗑️ 已刪除舊檔案: {file_path.name}")
            except Exception as e:
                print(f"⚠️ 刪除失敗 {file_path.name}: {e}")

        print(f"✅ 清理完成，保留最近 {keep_weeks} 周的檔案")

    def list_available_weeks(self) -> List[str]:
        """列出所有可用的周檔案"""
        if not self.base_dir.exists():
            return []

        files = sorted(self.base_dir.glob("*.json"))
        return [f.stem for f in files]

    # 私有輔助方法

    def _generate_search_queries(self) -> List[str]:
        """生成搜尋關鍵詞"""
        queries = []
        for keywords in self.search_categories.values():
            queries.extend(keywords[:2])  # 每類別取前2個
        return queries

    def _parse_week_filename(self, filename: str) -> tuple:
        """解析周檔案名（例如 2026_W10.json -> (2026, 10)）"""
        try:
            year_week = filename.replace(".json", "")
            year, week = year_week.split("_W")
            return int(year), int(week)
        except:
            raise ValueError(f"無效的周檔案名格式: {filename}")

    def _get_week_date_range(self, year: int, week: int) -> tuple:
        """獲取指定周的日期範圍"""
        # ISO 8601: 周一是一周的第一天
        jan_4 = datetime(year, 1, 4)
        week_one_monday = jan_4 - timedelta(days=jan_4.weekday())
        target_monday = week_one_monday + timedelta(weeks=week - 1)
        target_sunday = target_monday + timedelta(days=6)
        return target_monday, target_sunday


# 命令行接口
if __name__ == "__main__":
    import sys

    collector = WeeklyNewsCollector()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "collect":
            # 執行每日收集
            collector.collect_daily_news()

        elif command == "summary":
            # 顯示最近7天摘要
            summary = collector.get_recent_news_summary(days=7)
            print("\n" + "=" * 80)
            print("📊 最近 7 天新聞摘要")
            print("=" * 80)
            print(f"總新聞數: {summary['total_news']}")
            print(f"平均情緒: {summary['average_sentiment']:.2f}")
            print(f"\n各類別統計:")
            for category, stats in summary["by_category"].items():
                print(
                    f"  {category}: {stats['count']} 則, 情緒 {stats['avg_sentiment']:.2f}"
                )

        elif command == "cleanup":
            # 清理舊檔案
            collector.cleanup_old_files(keep_weeks=12)

        elif command == "list":
            # 列出所有周檔案
            weeks = collector.list_available_weeks()
            print(f"\n可用的周檔案 ({len(weeks)} 個):")
            for week in weeks:
                print(f"  - {week}")

        else:
            print("未知命令。可用命令: collect, summary, cleanup, list")

    else:
        # 預設執行每日收集
        print("執行每日新聞收集...")
        collector.collect_daily_news()
