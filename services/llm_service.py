"""
🧠 Gemini LLM 服務 (LLM Service) v1.0
負責生成真人等級的股票分析報告與智慧回覆
"""
import os
import google.generativeai as genai
import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

class LLMService:
    def __init__(self, api_key=None):
        # 安全地獲取 API Key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
        
        if not self.api_key:
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                self.api_key = None
                
        self.model = None
        self.model_name = "未選定"
        
        if self.api_key:
            try:
                # 診斷 API Key 狀態
                print(f"📡 正在使用 API Key 初始化 (長度: {len(self.api_key)})")
                genai.configure(api_key=self.api_key)
                
                # 🚀 智能模型偵測機制：解決 404 問題
                available_models = []
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            available_models.append(m.name)
                except Exception as e:
                    print(f"無法獲取模型清單: {e}")
                
                # 🚀 2026 智能降級策略：優先最強，Flash 保底確保可用性
                self.priority_list = [
                    "models/gemini-3.1-pro-preview", # 1. 巔峰推理 (若有額度)
                    "models/gemini-2.5-pro",         # 2. 穩定高階 (若有額度)
                    "models/gemini-2.5-flash",       # 3. 高速優質 (最佳平衡)
                    "models/gemini-2.0-flash",       # 4. 次世代快速
                ]
                
                self.available_models = available_models
                selected_model = None
                for p in self.priority_list:
                    if p in available_models:
                        selected_model = p
                        break
                
                # 如果優先名單都沒有，選第一個可用的
                if not selected_model and available_models:
                    selected_model = available_models[0]
                
                if selected_model:
                    self.model_name = selected_model
                    self.model = genai.GenerativeModel(selected_model)
                    print(f"✅ 已成功載入 Gemini 模型: {selected_model}")
                else:
                    print("❌ 找不到任何支援生成內容的 Gemini 模型")
                    
            except Exception as e:
                print(f"Gemini 配置失敗: {e}")
                self.model = None
        else:
            print("⚠️ 找不到 API Key，LLM 服務已停用")

    def call_gemini(self, prompt, use_fallback=True):
        """
        🚀 核心：具備指數退避重試與自動降級機制的 Gemini 調用
        """
        import time
        import random

        if not self.model:
            return "⚠️ Gemini API 未設定，無法提供服務。"

        # 基礎模型嘗試清單
        candidate_models = []
        current_model_idx = -1
        for i, m_name in enumerate(self.priority_list):
            if self.model_name == m_name:
                current_model_idx = i
                break
        
        if current_model_idx != -1:
            candidate_models = self.priority_list[current_model_idx:]
        else:
            candidate_models = [self.model_name] + [m for m in self.priority_list if "flash" in m]

        last_error = ""
        
        for model_name in candidate_models:
            if model_name not in self.available_models and model_name != self.model_name:
                continue
            
            # --- [優化] 指數退避重試機制 (針對單一模型嘗試 3 次) ---
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # 💡 每次請求前強制加入小間隔，降低發送速度 (1-2秒隨機)
                    time.sleep(random.uniform(1.0, 2.0))
                    
                    temp_model = genai.GenerativeModel(model_name)
                    response = temp_model.generate_content(prompt)
                    return response.text
                    
                except Exception as e:
                    error_msg = str(e)
                    last_error = error_msg
                    
                    # 💡 處理 429 Quota Exceeded 或 500/503 Service Unavailable
                    if "429" in error_msg or "quota" in error_msg.lower() or "500" in error_msg or "503" in error_msg:
                        wait_time = (2 ** retry) + random.uniform(1, 3) # 指數退避: 3, 5, 10 秒
                        print(f"⚠️ Gemini {model_name} 請求過快或額度不足 (嘗試 {retry+1}/{max_retries})，等待 {wait_time:.1f} 秒後重試...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 其他嚴重錯誤直接切換模型或報錯
                        print(f"❌ {model_name} 發生非配額錯誤: {error_msg}")
                        break # 跳出重試迴圈，嘗試下一個模型
            
            # 如果單一模型重試 3 次都失敗，且錯誤是配額問題，才切換到備援模型
            if "429" in last_error or "quota" in last_error.lower():
                print(f"🔄 模型 {model_name} 多次重試失敗，切換備援模型...")
                continue
            else:
                break # 若非配額問題，則不切換直接回傳錯誤
        
        return f"❌ 所有可用的 Gemini 模型調用皆失敗。最後錯誤: {last_error}"

    def generate_stock_analysis(self, stock_info, technical_data, fundamental_data, news_summary):
        """
        生成真人等級的股票分析 (已整合備援機制)
        """
        prompt = f"""
        你是一位資深的台股投資分析師，請針對以下數據進行深度診斷，並給出具體的投資建議。
        請用繁體中文，語氣要專業、客觀且像真人分析師。

        【基本資料】: {stock_info}
        【技術面分析】: {technical_data}
        【基本面數據】: {fundamental_data}
        【市場消息/情緒】: {news_summary}

        請依照以下結構分析：
        1. 總結當前走勢與多空力道。
        2. 潛在風險點與利多因素。
        3. 真人級別的投資建議（買入/賣出/觀望）及其理由。
        """
        return self.call_gemini(prompt)

    def extract_topic_features(self, news_text):
        """
        🚀 2026 進階優化：提取新聞實體 (股票代碼) 的精細情緒特徵
        這能讓 GNN 的每個節點 (股票) 獲得獨立且精確的新聞加權。
        """
        import json
        import re

        prompt = f"""
        請針對以下台股相關新聞，提取文中提到的「股票代碼」及其對應的情緒影響分數。
        
        【規則】：
        1. 僅提取具體的股票代碼 (例: 2330, 2454)。
        2. 情緒分數範圍: -1 (極大利空) 到 1 (極大利多)。
        3. 必須嚴格回傳 JSON 陣列格式，不得有其他贅字。
        
        【格式範例】：
        [
            {{"ticker": "2330", "sentiment": 0.8, "reason": "先進製程產能滿載"}},
            {{"ticker": "2454", "sentiment": -0.3, "reason": "競爭加劇導致市佔流失"}}
        ]
        
        【新聞內容】：
        {news_text}
        """
        
        raw_response = self.call_gemini(prompt)
        
        try:
            # 使用正則表達式尋找 JSON 陣列部分，避免 LLM 的說明文字干擾
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return parsed_data
            else:
                return [] # 找不到 JSON 則回傳空列表
        except Exception as e:
            print(f"⚠️ 解析實體情緒 JSON 失敗: {e}")
            return []

    def analyze_news_sentiment(self, news_text):
        """
        利用 LLM 進行深度新聞情緒分析 (已整合備援機制)
        """
        prompt = f"請分析以下新聞內容的情緒，給予 -1(極度利空) 到 1(極度利多) 之間的分數，並用一句話總結原因：\n\n{news_text}"
        return self.call_gemini(prompt)
