import requests
from bs4 import BeautifulSoup
import datetime
import time
import json
import os
from typing import Dict, List, Tuple
import hashlib

class NewsAnalyzer:
    def __init__(self):
        # 擴展的關鍵詞庫，帶權重系統
        # 權重: 3=強烈影響, 2=中等影響, 1=輕微影響
        self.positive_keywords = {
            # 強烈正面 (權重 3)
            "漲停": 3, "飆漲": 3, "暴漲": 3, "創新高": 3, "歷史新高": 3,
            "大爆發": 3, "狂飆": 3, "噴出": 3, "攻漲停": 3,
            
            # 中等正面 (權重 2)
            "上漲": 2, "強漲": 2, "上攻": 2, "攻高": 2, "走揚": 2,
            "大增": 2, "營收創高": 2, "獲利成長": 2, "業績亮眼": 2,
            "法說會利多": 2, "訂單滿載": 2, "產能滿載": 2, "展望樂觀": 2,
            "買進": 2, "加碼": 2, "調升目標價": 2, "升評": 2,
            
            # 輕微正面 (權重 1)
            "成長": 1, "獲利": 1, "營收": 1, "回升": 1, "反彈": 1,
            "看好": 1, "利多": 1, "強勢": 1, "發威": 1, "轉強": 1,
            "配息": 1, "殖利率": 1, "優於預期": 1, "熱門": 1, "題材": 1,
            "擴廠": 1, "投資": 1, "合作": 1, "接單": 1, "新訂單": 1,
            "外資買": 1, "法人買超": 1, "融資減": 1, "籌碼穩定": 1
        }
        
        self.negative_keywords = {
            # 強烈負面 (權重 -3)
            "跌停": -3, "重挫": -3, "崩跌": -3, "暴跌": -3, "創新低": -3,
            "腰斬": -3, "血崩": -3, "慘跌": -3, "閃崩": -3,
            
            # 中等負面 (權重 -2)
            "下跌": -2, "走跌": -2, "殺低": -2, "破底": -2, "失守": -2,
            "虧損": -2, "營收衰退": -2, "獲利下滑": -2, "展望保守": -2,
            "砍單": -2, "減產": -2, "裁員": -2, "虧損擴大": -2,
            "賣出": -2, "減碼": -2, "調降目標價": -2, "降評": -2,
            
            # 輕微負面 (權重 -1)
            "疲軟": -1, "弱勢": -1, "看淡": -1, "利空": -1, "不如預期": -1,
            "衰退": -1, "減少": -1, "失望": -1, "隱憂": -1, "壓力": -1,
            "庫存": -1, "競爭": -1, "風險": -1, "不確定": -1, "延後": -1,
            "外資賣": -1, "法人賣超": -1, "融資增": -1, "籌碼凌亂": -1
        }
        
        # 新聞來源可信度權重 (未來可擴展)
        self.source_weights = {
            "經濟日報": 1.2,
            "工商時報": 1.2,
            "鉅亨網": 1.1,
            "MoneyDJ": 1.1,
            "精實新聞": 1.1,
            "財訊": 1.0,
            "Yahoo": 1.0,
            "default": 1.0
        }
        
        # 快取機制（避免短時間內重複爬取）
        self.cache_dir = "data/news_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_duration = 1800  # 30分鐘快取
        
        # 🆕 產業關鍵詞映射（用於查詢國際產業新聞）
        self.industry_keywords = {
            "半導體": {
                "zh": ["半導體", "晶圓", "晶片", "IC設計", "封測"],
                "en": ["semiconductor", "chip shortage", "GPU demand", "AI chip", 
                      "memory shortage", "TSMC", "wafer", "foundry", "ASML",
                      "HBM memory", "DDR5", "chiplet", "3nm process"]
            },
            "記憶體": {
                "zh": ["記憶體", "DRAM", "NAND Flash", "儲存"],
                "en": ["memory shortage", "DRAM price", "NAND flash", "HBM",
                      "memory demand", "storage", "SK Hynix", "Micron"]
            },
            "AI": {
                "zh": ["人工智慧", "AI晶片", "機器學習", "深度學習"],
                "en": ["AI chip demand", "GPU shortage", "NVIDIA", "AI server",
                      "ChatGPT", "generative AI", "AI accelerator", "inference chip"]
            },
            "光電": {
                "zh": ["光學", "鏡頭", "面板", "顯示器"],
                "en": ["optical lens", "camera module", "display panel", "OLED",
                      "mini LED", "micro LED", "LCD"]
            },
            "電動車": {
                "zh": ["電動車", "電池", "充電樁", "車用晶片"],
                "en": ["EV battery", "electric vehicle", "Tesla", "BYD",
                      "automotive chip", "battery shortage", "charging station"]
            },
            "5G": {
                "zh": ["5G", "通訊", "基地台", "射頻"],
                "en": ["5G deployment", "telecom equipment", "RF chip",
                      "base station", "Qualcomm", "MediaTek"]
            },
            "綠能": {
                "zh": ["太陽能", "風電", "儲能", "電池"],
                "en": ["solar energy", "wind power", "energy storage",
                      "renewable energy", "green energy", "battery"]
            },
            "生技": {
                "zh": ["生技", "製藥", "疫苗", "醫療器材"],
                "en": ["biotech", "pharmaceutical", "vaccine", "medical device",
                      "FDA approval", "clinical trial"]
            }
        }
    
    def get_sentiment_emoji(self, score):
        """獲取情感表情符號"""
        if score >= 5: return "🚀"  # Extremely positive
        if score >= 2: return "🔥"  # Very positive
        if score > 0: return "🙂"   # Positive
        if score == 0: return "😐"  # Neutral
        if score > -2: return "🙁"  # Negative
        if score > -5: return "😰"  # Very negative
        return "😭"  # Extremely negative
    
    def get_sentiment_level(self, score):
        """獲取情感等級描述"""
        if score >= 5: return "極度樂觀"
        if score >= 2: return "明顯正面"
        if score > 0: return "略為正面"
        if score == 0: return "中性"
        if score > -2: return "略為負面"
        if score > -5: return "明顯負面"
        return "極度悲觀"
    
    def _get_cache_path(self, ticker):
        """獲取快取文件路徑"""
        cache_key = hashlib.md5(ticker.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{ticker}_{cache_key}.json")
    
    def _load_cache(self, ticker):
        """載入快取的新聞"""
        cache_path = self._get_cache_path(ticker)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 檢查快取是否過期
            cache_time = datetime.datetime.fromisoformat(cache_data['timestamp'])
            now = datetime.datetime.now()
            
            if (now - cache_time).total_seconds() < self.cache_duration:
                return cache_data['news']
        except:
            pass
        
        return None
    
    def _save_cache(self, ticker, news_list):
        """保存新聞到快取"""
        cache_path = self._get_cache_path(ticker)
        try:
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'news': news_list
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _calculate_time_weight(self, news_date_str):
        """計算時間衰減權重（新聞越新，權重越高）"""
        try:
            news_date = datetime.datetime.strptime(news_date_str, "%Y-%m-%d").date()
            today = datetime.date.today()
            days_ago = (today - news_date).days
            
            if days_ago == 0: return 1.0      # 今天的新聞：100%權重
            if days_ago == 1: return 0.8      # 昨天：80%
            if days_ago <= 3: return 0.6      # 3天內：60%
            if days_ago <= 7: return 0.4      # 1週內：40%
            return 0.2                         # 超過1週：20%
        except:
            return 0.5  # 無法解析日期，給予中等權重
    
    def _detect_source(self, title, link):
        """偵測新聞來源"""
        for source, weight in self.source_weights.items():
            if source.lower() in link.lower() or source in title:
                return source, weight
        return "未知", self.source_weights["default"]

    def get_stock_news(self, ticker, max_retries=2):
        """
        從 Yahoo 奇摩股市抓取該股票的新聞標題
        ticker: 股票代碼 (例如 "2330")
        max_retries: 最大重試次數
        """
        # 先檢查快取
        cached_news = self._load_cache(ticker)
        if cached_news is not None:
            return cached_news
        
        # 移除 .TW 或 .TWO 後綴
        stock_id = ticker.split('.')[0]
        
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive"
        }
        
        news_list = []
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 嘗試多種選擇器以提高成功率
                    selectors = [
                        'a[class*="Fw(b)"][class*="Fz(16px)"]',  # Yahoo 當前版本
                        'h3 a',                                    # 通用標題連結
                        'a[class*="title"]',                       # 包含 title 的連結
                        '.StreamMegaItem a',                       # Stream 項目
                        'li a[href*="/news/"]'                     # 新聞列表項目
                    ]
                    
                    links = []
                    for selector in selectors:
                        links = soup.select(selector)
                        if links:
                            break
                    
                    count = 0
                    seen_titles = set()  # 避免重複新聞
                    
                    for link in links:
                        title = link.get_text().strip()
                        href = link.get('href', '')
                        
                        # 過濾條件
                        if len(title) < 10 or title in seen_titles:
                            continue
                        if not href or 'news' not in href.lower():
                            continue
                        
                        # 完整URL
                        if not href.startswith('http'):
                            href = 'https://tw.stock.yahoo.com' + href
                        
                        seen_titles.add(title)
                        news_list.append({
                            "title": title,
                            "link": href,
                            "date": datetime.date.today().strftime("%Y-%m-%d")
                        })
                        count += 1
                        
                        if count >= 15:  # 最多取15則
                            break
                    
                    if news_list:
                        # 保存到快取
                        self._save_cache(ticker, news_list)
                        break
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    time.sleep(1)  # 等待1秒後重試
                    continue
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    # 最後一次失敗，返回空列表
                    pass
        
        return news_list

    def analyze_sentiment(self, ticker, detailed=False):
        """
        分析新聞情緒分數（升級版：含權重、時間衰減、可信度評估）
        
        Args:
            ticker: 股票代碼
            detailed: 是否返回詳細分析
        
        Returns:
            (score, news_list) 或 (score, news_list, detail_dict) if detailed=True
            score: 綜合情感分數 (>0 正面, <0 負面)
            news_list: 格式化的新聞列表
            detail_dict: 詳細分析數據（僅在 detailed=True 時返回）
        """
        news_list = self.get_stock_news(ticker)
        
        if not news_list:
            if detailed:
                return 0, [], {"error": "無法獲取新聞"}
            return 0, []
        
        total_score = 0.0
        weighted_score = 0.0
        total_weight = 0.0
        details = []
        sentiment_breakdown = {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total_news": len(news_list),
            "avg_score": 0,
            "confidence": 0,
            "time_weighted_score": 0
        }
        
        for news in news_list:
            title = news['title']
            news_date = news['date']
            link = news.get('link', '')
            
            # 計算關鍵詞分數（帶權重）
            keyword_score = 0
            matched_keywords = []
            
            # 正面關鍵詞檢測
            for keyword, weight in self.positive_keywords.items():
                if keyword in title:
                    keyword_score += weight
                    matched_keywords.append(f"+{keyword}({weight})")
            
            # 負面關鍵詞檢測
            for keyword, weight in self.negative_keywords.items():
                if keyword in title:
                    keyword_score += weight  # weight 已經是負數
                    matched_keywords.append(f"{keyword}({weight})")
            
            # 偵測新聞來源並應用可信度權重
            source, source_weight = self._detect_source(title, link)
            
            # 計算時間衰減權重
            time_weight = self._calculate_time_weight(news_date)
            
            # 綜合權重 = 關鍵詞分數 × 來源可信度 × 時間權重
            final_weight = source_weight * time_weight
            weighted_impact = keyword_score * final_weight
            
            total_score += keyword_score
            weighted_score += weighted_impact
            total_weight += final_weight
            
            # 統計分類
            if keyword_score > 0:
                sentiment_breakdown["positive_count"] += 1
            elif keyword_score < 0:
                sentiment_breakdown["negative_count"] += 1
            else:
                sentiment_breakdown["neutral_count"] += 1
            
            # 格式化輸出
            if keyword_score != 0:
                emoji = "🔴" if keyword_score > 0 else "🟢"  # 台股紅漲綠跌
                intensity = abs(keyword_score)
                
                if intensity >= 5:
                    level = "【極強】"
                elif intensity >= 3:
                    level = "【強】"
                elif intensity >= 2:
                    level = "【中】"
                else:
                    level = ""
                
                detail_str = f"{emoji}{level} {title}"
                if matched_keywords and detailed:
                    detail_str += f" [{','.join(matched_keywords[:3])}]"
                
                details.append(detail_str)
            else:
                details.append(f"⚪ {title}")
        
        # 計算最終分數（使用加權平均）
        if total_weight > 0:
            final_score = round(weighted_score / total_weight, 2)
        else:
            final_score = 0
        
        # 計算可信度（基於新聞數量和一致性）
        if len(news_list) >= 10:
            quantity_factor = 1.0
        elif len(news_list) >= 5:
            quantity_factor = 0.8
        else:
            quantity_factor = 0.6
        
        # 情感一致性（正面或負面占比越高，可信度越高）
        if len(news_list) > 0:
            consistency = max(
                sentiment_breakdown["positive_count"],
                sentiment_breakdown["negative_count"]
            ) / len(news_list)
        else:
            consistency = 0
        
        confidence = round(quantity_factor * consistency * 100, 1)
        
        # 補充統計資訊
        sentiment_breakdown["avg_score"] = round(total_score / len(news_list), 2) if news_list else 0
        sentiment_breakdown["weighted_score"] = final_score
        sentiment_breakdown["confidence"] = confidence
        sentiment_breakdown["sentiment_level"] = self.get_sentiment_level(final_score)
        
        if detailed:
            return int(round(final_score)), details, sentiment_breakdown
        
        return int(round(final_score)), details
    
    def batch_analyze(self, tickers: List[str], top_n=10):
        """
        批量分析多支股票的新聞情感
        
        Args:
            tickers: 股票代碼列表
            top_n: 返回排名前N的股票
        
        Returns:
            排序後的分析結果列表
        """
        results = []
        
        for ticker in tickers:
            score, _, details = self.analyze_sentiment(ticker, detailed=True)
            
            if not isinstance(details, dict):
                continue
            
            results.append({
                'ticker': ticker,
                'score': score,
                'emoji': self.get_sentiment_emoji(score),
                'level': details.get('sentiment_level', '中性'),
                'confidence': details.get('confidence', 0),
                'news_count': details.get('total_news', 0),
                'positive': details.get('positive_count', 0),
                'negative': details.get('negative_count', 0)
            })
        
        # 按分數排序
        results.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
        
        return results[:top_n]
    
    def get_market_sentiment(self, tickers: List[str]) -> Dict:
        """
        分析整體市場情緒
        
        Args:
            tickers: 代表性股票列表（如權值股）
        
        Returns:
            市場整體情感統計
        """
        all_scores = []
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        
        for ticker in tickers:
            score, _, details = self.analyze_sentiment(ticker, detailed=True)
            
            if isinstance(details, dict):
                all_scores.append(score)
                total_positive += details.get('positive_count', 0)
                total_negative += details.get('negative_count', 0)
                total_neutral += details.get('neutral_count', 0)
        
        if not all_scores:
            return {
                'market_score': 0,
                'market_sentiment': '無法判斷',
                'positive_ratio': 0,
                'negative_ratio': 0
            }
        
        market_score = sum(all_scores) / len(all_scores)
        total_news = total_positive + total_negative + total_neutral
        
        return {
            'market_score': round(market_score, 2),
            'market_sentiment': self.get_sentiment_level(market_score),
            'market_emoji': self.get_sentiment_emoji(market_score),
            'positive_ratio': round(total_positive / total_news * 100, 1) if total_news > 0 else 0,
            'negative_ratio': round(total_negative / total_news * 100, 1) if total_news > 0 else 0,
            'neutral_ratio': round(total_neutral / total_news * 100, 1) if total_news > 0 else 0,
            'total_news_analyzed': total_news,
            'stocks_analyzed': len(tickers)
        }
    
    def clear_cache(self):
        """清除所有新聞快取"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
            return True
        except:
            return False
    
    def get_cache_stats(self):
        """獲取快取統計資訊"""
        if not os.path.exists(self.cache_dir):
            return {'total_cached': 0, 'total_size': 0}
        
        files = os.listdir(self.cache_dir)
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in files
        )
        
        return {
            'total_cached': len(files),
            'total_size_kb': round(total_size / 1024, 2),
            'cache_dir': self.cache_dir
        }
    
    def detect_industry(self, ticker, stock_name=""):
        """自動偵測股票所屬產業
        
        Args:
            ticker: 股票代碼
            stock_name: 股票名稱（選填）
        
        Returns:
            產業名稱（如："半導體"、"AI" 等）或 None
        """
        # 1. 優先嘗試從 tw_stock_list.csv 讀取精確產業
        try:
            # 移除 .TW 或 .TWO 後綴以便比對
            code = ticker.split('.')[0]
            
            # 尋找 CSV 檔案 (嘗試多個可能路徑)
            paths = ["data/tw_stock_list.csv", "tw_stock_list.csv"]
            for path in paths:
                if os.path.exists(path):
                    import pandas as pd
                    df = pd.read_csv(path)
                    # 確保代碼欄位存在且為字串
                    if '代碼' in df.columns:
                        df['代碼'] = df['代碼'].astype(str)
                        row = df[df['代碼'] == code]
                        if not row.empty and '產業' in row.columns:
                            exact_industry = row.iloc[0]['產業']
                            
                            # 映射到我們支援的 8 大國際產業分類
                            mapping = {
                                "半導體業": "半導體",
                                "電腦及週邊設備業": "AI",
                                "通信網路業": "5G",
                                "電子零組件業": "AI",
                                "光電業": "光電",
                                "其他電子業": "AI",
                                "電子通路業": "AI",
                                "資訊服務業": "AI",
                                "生技醫療業": "生技",
                                "電機機械": "電動車", # 許多電機股轉向電動車
                                "汽車工業": "電動車",
                                "油電燃氣業": "綠能"
                            }
                            
                            if exact_industry in mapping:
                                return mapping[exact_industry]
                            
                            # 如果是 ETF
                            if exact_industry == "ETF":
                                # 根據名稱猜測 ETF 類型
                                if "半導體" in stock_name: return "半導體"
                                if "AI" in stock_name or "科技" in stock_name: return "AI"
                                if "綠能" in stock_name or "能源" in stock_name: return "綠能"
                                if "生技" in stock_name: return "生技"
                                return "AI" # 預設 AI (熱門 ETF 多為科技類)
                            
                            # 如果是其他產業，回傳名稱本身，後續嘗試關鍵詞匹配
                            break
        except Exception:
            pass

        # 2. 既有的寫死映射 (作為快速備案)
        industry_mapping = {
            "2330": "半導體", "2454": "半導體", "2379": "半導體",
            "3034": "半導體", "3661": "半導體", "2303": "半導體",
            "6770": "半導體", "3711": "記憶體", "2337": "記憶體",
            "2344": "記憶體", "3008": "光電", "2409": "光電",
            "2301": "光電", "2382": "AI", "2357": "AI",
            "6269": "AI", "2308": "電動車", "1513": "電動車",
            "2327": "電動車", "1319": "電動車", "1503": "電動車"
        }
        
        code = ticker.split('.')[0]
        if code in industry_mapping:
            return industry_mapping[code]
        
        # 3. 根據股票名稱或產業名稱關鍵詞判斷
        name_to_check = f"{stock_name} {ticker}"
        
        name_keywords = {
            "半導體": ["半導體", "IC", "晶圓", "晶片", "矽", "台積", "聯發"],
            "記憶體": ["記憶體", "DRAM", "快閃", "旺宏", "華邦"],
            "AI": ["AI", "伺服器", "雲端", "廣達", "技嘉", "微星", "緯創"],
            "光電": ["光", "面板", "顯示", "友達", "群創", "大立光"],
            "電動車": ["電動", "電池", "車用", "特斯拉", "充電", "鴻海"],
            "5G": ["5G", "通訊", "射頻", "中華電", "遠傳", "台哥大"],
            "綠能": ["太陽能", "風電", "儲能", "能源", "綠能", "氫"],
            "生技": ["生技", "製藥", "醫", "疫苗", "新藥"]
        }
        
        for industry, keywords in name_keywords.items():
            for keyword in keywords:
                if keyword in name_to_check:
                    return industry
        
        return None
    
    def get_industry_news(self, industry, language="en", max_results=10):
        """查詢產業國際新聞（使用 Google News RSS）
        
        Args:
            industry: 產業名稱（如："半導體"、"AI"）
            language: 語言（"zh" 中文, "en" 英文）
            max_results: 最多返回幾則新聞
        
        Returns:
            新聞列表
        """
        if industry not in self.industry_keywords:
            return []
        
        # 檢查快取
        cache_key = f"industry_{industry}_{language}"
        cached_news = self._load_cache(cache_key)
        if cached_news is not None:
            return cached_news[:max_results]
        
        keywords = self.industry_keywords[industry].get(language, [])
        if not keywords:
            return []
        
        news_list = []
        
        # 使用前3個關鍵詞查詢（避免過多請求）
        for keyword in keywords[:3]:
            try:
                # Google News RSS Feed
                search_query = keyword.replace(" ", "+")
                
                if language == "zh":
                    url = f"https://news.google.com/rss/search?q={search_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
                else:
                    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')
                    
                    for item in items[:5]:  # 每個關鍵詞取5則
                        title = item.find('title')
                        link = item.find('link')
                        pub_date = item.find('pubDate')
                        
                        if title and link:
                            # 解析日期
                            try:
                                if pub_date:
                                    date_str = datetime.datetime.strptime(
                                        pub_date.text, 
                                        "%a, %d %b %Y %H:%M:%S %Z"
                                    ).strftime("%Y-%m-%d")
                                else:
                                    date_str = datetime.date.today().strftime("%Y-%m-%d")
                            except:
                                date_str = datetime.date.today().strftime("%Y-%m-%d")
                            
                            news_list.append({
                                "title": title.text.strip(),
                                "link": link.text.strip(),
                                "date": date_str,
                                "keyword": keyword,
                                "industry": industry
                            })
                
                time.sleep(0.5)  # 避免請求過快
                
            except Exception as e:
                continue
        
        # 去重（根據標題）
        seen_titles = set()
        unique_news = []
        for news in news_list:
            title = news['title']
            if title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        # 保存快取
        if unique_news:
            self._save_cache(cache_key, unique_news)
        
        return unique_news[:max_results]
    
    def analyze_industry_sentiment(self, industry, detailed=False):
        """分析產業新聞情緒
        
        Args:
            industry: 產業名稱
            detailed: 是否返回詳細資訊
        
        Returns:
            (score, summary) 或更詳細的分析結果
        """
        # 同時查詢中英文新聞
        zh_news = self.get_industry_news(industry, "zh", max_results=10)
        en_news = self.get_industry_news(industry, "en", max_results=10)
        
        all_news = zh_news + en_news
        
        if not all_news:
            if detailed:
                return 0, "無法獲取產業新聞", {"total_news": 0}
            return 0, "無法獲取產業新聞"
        
        # 擴展的產業關鍵詞（英文）
        industry_positive_keywords = {
            "shortage": 2,      # 缺貨（對供應商是好事）
            "demand surge": 3,  # 需求激增
            "strong demand": 2, # 需求強勁
            "price increase": 2,# 價格上漲
            "capacity expansion": 2, # 產能擴張
            "record high": 3,   # 創新高
            "breakthrough": 2,  # 突破
            "partnership": 1,   # 合作
            "investment": 1,    # 投資
            "growth": 1,        # 成長
            "booming": 3,       # 蓬勃發展
            "surging": 2        # 激增
        }
        
        industry_negative_keywords = {
            "oversupply": -2,   # 供過於求
            "declining": -2,    # 下滑
            "slowdown": -2,     # 放緩
            "weak demand": -2,  # 需求疲軟
            "price drop": -2,   # 價格下跌
            "layoff": -2,       # 裁員
            "delay": -1,        # 延遲
            "concern": -1,      # 擔憂
            "risk": -1          # 風險
        }
        
        total_score = 0
        matched_news = []
        
        for news in all_news:
            title = news['title'].lower()
            news_score = 0
            matched_keywords = []
            
            # 檢查正面關鍵詞
            for keyword, weight in industry_positive_keywords.items():
                if keyword in title:
                    news_score += weight
                    matched_keywords.append(f"+{keyword}")
            
            # 檢查負面關鍵詞
            for keyword, weight in industry_negative_keywords.items():
                if keyword in title:
                    news_score += weight
                    matched_keywords.append(f"{keyword}")
            
            # 也檢查原有的中文關鍵詞
            for keyword, weight in self.positive_keywords.items():
                if keyword in news['title']:
                    news_score += weight
                    matched_keywords.append(f"+{keyword}")
            
            for keyword, weight in self.negative_keywords.items():
                if keyword in news['title']:
                    news_score += weight
                    matched_keywords.append(f"{keyword}")
            
            total_score += news_score
            
            if news_score != 0 or len(matched_news) < 10:
                emoji = "🔴" if news_score > 0 else ("🟢" if news_score < 0 else "⚪")
                matched_news.append({
                    "title": news['title'],
                    "score": news_score,
                    "date": news['date'],
                    "emoji": emoji,
                    "keywords": matched_keywords[:3]
                })
        
        avg_score = round(total_score / len(all_news), 2) if all_news else 0
        
        summary = f"{industry}產業情緒: {self.get_sentiment_level(avg_score)} ({avg_score:.1f}分)"
        summary += f" | 分析 {len(all_news)} 則國際新聞"
        
        if detailed:
            return avg_score, summary, {
                "total_news": len(all_news),
                "zh_news_count": len(zh_news),
                "en_news_count": len(en_news),
                "avg_score": avg_score,
                "sentiment_level": self.get_sentiment_level(avg_score),
                "matched_news": matched_news[:15]
            }
        
        return avg_score, summary
