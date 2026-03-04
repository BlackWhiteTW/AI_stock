import os
import sys
from dotenv import load_dotenv
load_dotenv()

# 確保能引用到服務層
sys.path.append(os.getcwd())

from services.llm_service import LLMService

def test_top_tier_model():
    print("--- 🚀 正在啟動頂級 Gemini 模型測試 ---")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        print("❌ 錯誤: 腳本環境中找不到 API Key")
    else:
        print(f"📡 已偵測到 API Key (長度: {len(api_key)})")

    llm = LLMService()
    
    print(f"✅ 目前啟動的模型: {llm.model_name}")
    
    # 模擬台積電 (2330) 的複雜數據情境
    stock_info = "台積電 (2330.TW) 目前股價: 1,150 TWD"
    technical_data = "日線趨勢: 多頭排列 (Uptrend), RSI: 68 (高檔偏強), MFI: 72 (資金持續流入), 20MA/60MA 呈黃金交叉。"
    fundamental_data = "2月營收年增率: 28.5%, 毛利率: 53.5%, 2026 預估 EPS: 45.8 TWD, 財務健康度評分: 92/100。"
    news_summary = """
    1. 蘋果與輝達追加 2nm 訂單，產能利用率預計 2026 全年滿載。
    2. 地緣政治風險：美國商務部加強對先進製程出口限制。
    3. 市場情緒：散戶與外資同步偏多，投信近日獲利了結。
    """
    
    print("\n⏳ 正在生成真人級別深度報告...\n")
    report = llm.generate_stock_analysis(stock_info, technical_data, fundamental_data, news_summary)
    
    print("="*50)
    print(report)
    print("="*50)

if __name__ == "__main__":
    test_top_tier_model()
