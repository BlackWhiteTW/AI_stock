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
                priority_list = [
                    "models/gemini-3.1-pro-preview", # 1. 巔峰推理 (若有額度)
                    "models/gemini-2.5-pro",         # 2. 穩定高階 (若有額度)
                    "models/gemini-2.5-flash",       # 3. 高速優質 (最佳平衡)
                    "models/gemini-2.0-flash",       # 4. 次世代快速
                ]
                
                selected_model = None
                for p in priority_list:
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

    def generate_stock_analysis(self, stock_info, technical_data, fundamental_data, news_summary):
        """
        生成真人等級的股票分析
        """
        if not self.model:
            return "⚠️ Gemini API Key 未設定，無法生成智慧分析報告。"

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

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ 生成報告時發生錯誤: {str(e)}"

    def analyze_news_sentiment(self, news_text):
        """
        利用 LLM 進行深度新聞情緒分析
        """
        if not self.model:
            return None
            
        # 使用多行字串避免斷行錯誤
        prompt = f"請分析以下新聞內容的情緒，給予 -1(極度利空) 到 1(極度利多) 之間的分數，並用一句話總結原因：\n\n{news_text}"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except:
            return None
