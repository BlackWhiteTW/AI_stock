"""
🌐 SerpAPI 搜尋引擎服務 (Search Engine Service) v1.0
負責從 Google News 抓取即時的市場消息與網路討論
"""
import os
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services.quota_manager import QuotaManager

class SearchEngine:
    def __init__(self, api_key=None, cache_dir="data/news_cache"):
        self.api_key = api_key or os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY")
        self.base_url = "https://serpapi.com/search"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_days = 7
        self.quota_manager = QuotaManager() 

    def get_latest_news(self, query, num_results=5):
        """
        獲取搜尋結果，優先使用 7 天內的本地快取
        """
        # 1. 建立快取檔名 (使用簡單的雜湊處理 query)
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_filename = f"serp_{query_hash}.json"
        cache_path = self.cache_dir / cache_filename

        # 2. 檢查是否有有效的快取
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cached_data.get("timestamp"))
                if datetime.now() - cache_time < timedelta(days=self.cache_ttl_days):
                    print(f"♻️ 使用快取新聞 (搜尋詞: {query})")
                    return cached_data.get("content")
            except:
                pass

        # 3. 如果沒快取或已過期，才調用 SerpAPI
        if not self.api_key:
            return "⚠️ SerpAPI Key 未設定，且無有效快取。"

        # 記錄請求額度
        self.quota_manager.log_request()
        print(f"🌐 正在調用 SerpAPI 獲取新資訊... (搜尋詞: {query}, 剩餘預估: {self.quota_manager.get_status()['remaining']})")
        
        params = {
            "engine": "google_news",
            "q": query,
            "gl": "tw",
            "hl": "zh-tw",
            "api_key": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if "error" in data:
                # 如果 API 報錯，嘗試回傳過期的快取作為備案
                if cache_path.exists():
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        old_data = json.load(f)
                    return f"【⚠️ API 額度用盡，顯示舊聞】\n{old_data.get('content')}"
                return f"❌ SerpAPI 錯誤: {data['error']}"
            
            news_results = data.get("news_results", [])
            if not news_results:
                return "目前查無相關即時新聞。"

            summaries = []
            for news in news_results[:num_results]:
                title = news.get("title", "無標題")
                source = "未知來源"
                if "source" in news:
                    if isinstance(news["source"], dict):
                        source = news["source"].get("name", "未知來源")
                    else:
                        source = str(news["source"])
                
                date = news.get("date", "近期")
                summaries.append(f"【{source}】{title} ({date})")
            
            content = "\n\n".join(summaries)

            # 4. 儲存結果到快取
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "content": content
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            return content
            
        except Exception as e:
            # 發生網路錯誤時，同樣嘗試使用舊快取
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                return old_data.get('content')
            return f"❌ 搜尋服務異常: {str(e)}"
