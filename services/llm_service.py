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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # 如果環境變數沒抓到，再嘗試從 st.secrets 抓取 (避免報錯)
        if not self.api_key:
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                self.api_key = None
                
        self.model_name = "gemini-1.5-flash"
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Gemini 配置失敗: {e}")
                self.model = None
        else:
            self.model = None

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
