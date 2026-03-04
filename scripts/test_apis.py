import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini():
    print("--- 正在測試 Gemini API ---")
    load_dotenv()
    # 支援您 .env 中的兩種可能名稱
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    
    if not api_key:
        print("❌ 錯誤: 找不到 API Key (請檢查 .env 是否有 GEMINI_API_KEY 或 GOOGLE_GEMINI_API_KEY)")
        return False
    
    try:
        genai.configure(api_key=api_key)
        # 2026 更新: 使用最新通用的 flash 名稱
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("這是一次連線測試。")
        print(f"✅ Gemini 連線成功！")
        return True
    except Exception as e:
        print(f"❌ Gemini 測試失敗: {e}")
        return False

def test_serpapi():
    print("\n--- 正在測試 SerpAPI ---")
    load_dotenv()
    # 支援您 .env 中的兩種可能名稱
    api_key = os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY")
    
    if not api_key:
        print("❌ 錯誤: 找不到 API Key (請檢查 .env 是否有 SERPAPI_KEY 或 SERP_API_KEY)")
        return False
    
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_news",
        "q": "台股",
        "api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if "error" in data:
            print(f"❌ SerpAPI 報錯: {data['error']}")
            return False
        print(f"✅ SerpAPI 連線成功！")
        return True
    except Exception as e:
        print(f"❌ SerpAPI 測試失敗: {e}")
        return False

if __name__ == "__main__":
    g_ok = test_gemini()
    s_ok = test_serpapi()
    print("\n" + "="*30)
    print(f"測試結果: Gemini [{'OK' if g_ok else 'FAIL'}] | SerpAPI [{'OK' if s_ok else 'FAIL'}]")
    print("="*30)
