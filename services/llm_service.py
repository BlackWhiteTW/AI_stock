"""
🧠 Gemini LLM 服務 (LLM Service) v1.0
負責生成真人等級的股票分析報告與智慧回覆
"""

import os
import time
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
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
        )

        if not self.api_key:
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                self.api_key = None

        self.model = None
        self.model_name = "未選定"
        self.available_models = []
        self.priority_list = [
            "models/gemini-3.1-flash-lite-preview",
        ]

    def _ensure_model_initialized(self):
        """按需初始化模型，避免啟動時阻塞 UI"""
        if self.model:
            return True

        if not self.api_key:
            print("⚠️ 找不到 API Key，LLM 服務已停用")
            return False

        try:
            # 診斷 API Key 狀態
            print(f"📡 正在使用 API Key 初始化 (長度: {len(self.api_key)})")
            genai.configure(api_key=self.api_key)

            # 🚀 智能模型偵測機制：解決 404 問題
            available_models = []
            try:
                print("📡 正在獲取可用模型清單...")
                list_start = time.time()
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        available_models.append(m.name)
                print(
                    f"✅ 已獲取模型清單 (耗時: {time.time() - list_start:.2f}s, 共 {len(available_models)} 個)"
                )
            except Exception as e:
                print(f"無法獲取模型清單: {e}")
                # 保底
                available_models = [
                    "models/gemini-3.1-flash-lite-preview",
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
                return True
            else:
                print("❌ 找不到任何支援生成內容的 Gemini 模型")
                return False

        except Exception as e:
            print(f"Gemini 配置失敗: {e}")
            return False

    def call_gemini(self, prompt, use_fallback=True):
        """
        🚀 核心：具備指數退避重試與自動降級機制的 Gemini 調用 (已升級至 generate_content API)
        """
        import time
        import random

        if not self._ensure_model_initialized():
            return "⚠️ Gemini API 未設定，無法提供服務。"
        # 重試設定
        max_attempts = 3
        base_backoff = 1.0
        last_error = None

        # 預設生成參數（保守且通用）
        gen_params = {
            "max_output_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.95,
        }

        def _extract_text(resp):
            try:
                # GenerativeAI 回應對象
                if hasattr(resp, "text"):
                    return getattr(resp, "text")

                # dict-like responses
                if isinstance(resp, dict):
                    if "candidates" in resp and resp["candidates"]:
                        c = resp["candidates"][0]
                        if isinstance(c, dict):
                            content = c.get("content")
                            if isinstance(content, dict):
                                return content.get("parts", [{}])[0].get("text", "")
                            return c.get("text") or str(c)
                    if "text" in resp:
                        return resp.get("text")
                    if "content" in resp:
                        return resp.get("content")
                    return str(resp)

                # object-like responses
                if hasattr(resp, "content"):
                    return getattr(resp, "content")
                if hasattr(resp, "candidates") and getattr(resp, "candidates"):
                    cand = getattr(resp, "candidates")[0]
                    if hasattr(cand, "content"):
                        content_obj = getattr(cand, "content")
                        if hasattr(content_obj, "parts"):
                            parts = getattr(content_obj, "parts")
                            if parts and hasattr(parts[0], "text"):
                                return getattr(parts[0], "text")
                    return str(cand)
            except Exception:
                pass
            return str(resp)

        for attempt in range(1, max_attempts + 1):
            try:
                # 優先使用最新的 generate_content API
                resp = None
                try:
                    if self.model is not None:
                        if hasattr(self.model, "generate_content"):
                            # 新 API：generate_content
                            resp = self.model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    **gen_params
                                ),
                                stream=False,
                            )
                        elif hasattr(self.model, "generate"):
                            # 舊 API：generate (fallback)
                            resp = self.model.generate(prompt=prompt, **gen_params)
                        else:
                            raise Exception("Model does not support generate methods")
                    else:
                        raise Exception("Model not initialized")
                except (TypeError, AttributeError):
                    # 某些版本使用不同參數名稱
                    try:
                        if self.model is not None and hasattr(
                            self.model, "generate_content"
                        ):
                            resp = self.model.generate_content(prompt)
                        elif self.model is not None and hasattr(self.model, "generate"):
                            resp = self.model.generate(prompt=prompt)
                        else:
                            resp = genai.generate(model=self.model_name, input=prompt)
                    except Exception as e:
                        raise

                text = _extract_text(resp)
                if text and len(str(text).strip()) > 0:
                    return str(text)
                last_error = "empty response"
            except Exception as e:
                last_error = str(e)

            # 指數退避
            sleep_for = base_backoff * (2 ** (attempt - 1)) + random.random() * 0.2
            time.sleep(sleep_for)

        return f"❌ 所有可用的 Gemini 模型調用皆失敗。最後錯誤: {last_error}"

    def generate_stock_analysis(
        self, stock_info, technical_data, fundamental_data, news_summary
    ):
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
            json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return parsed_data
            else:
                return []  # 找不到 JSON 則回傳空列表
        except Exception as e:
            print(f"⚠️ 解析實體情緒 JSON 失敗: {e}")
            return []

    def analyze_news_sentiment(self, news_text):
        """
        利用 LLM 進行深度新聞情緒分析 (已整合備援機制)
        """
        prompt = f"請分析以下新聞內容的情緒，給予 -1(極度利空) 到 1(極度利多) 之間的分數，並用一句話總結原因：\n\n{news_text}"
        return self.call_gemini(prompt)
