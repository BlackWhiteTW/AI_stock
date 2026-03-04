import os
import google.generativeai as genai
from dotenv import load_dotenv

def list_available_models():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    
    if not api_key:
        print("❌ 錯誤: 找不到 API Key")
        return
    
    genai.configure(api_key=api_key)
    
    print("--- 正在獲取可用模型清單 ---")
    try:
        models = genai.list_models()
        count = 0
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ 模型: {m.name} | 支援: generateContent")
                count += 1
            else:
                print(f"⚪ 模型: {m.name} | 支援: {m.supported_generation_methods}")
        
        if count == 0:
            print("❌ 警告: 沒有任何模型支援 generateContent")
    except Exception as e:
        print(f"❌ 無法連線或驗證 API: {e}")

if __name__ == "__main__":
    list_available_models()
