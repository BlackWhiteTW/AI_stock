import google.generativeai as genai

# 模擬台積電數據
stock_info = "台積電 (2330.TW) 目前股價: 1,150 TWD"
technical_data = "日線趨勢: 多頭排列 (Uptrend), RSI: 68 (高檔偏強), MFI: 72 (資金持續流入), 20MA/60MA 呈黃金交叉。"
fundamental_data = "2月營收年增率: 28.5%, 毛利率: 53.5%, 2026 預估 EPS: 45.8 TWD, 財務健康度評分: 92/100。"
news_summary = """
1. 蘋果與輝達追加 2nm 訂單，產能利用率預計 2026 全年滿載。
2. 地緣政治風險：美國商務部加強對先進製程出口限制。
3. 市場情緒：散戶與外資同步偏多，投信近日獲利了結。
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

# 直接設定 Key 並執行
try:
    genai.configure(api_key="AIzaSyDQxQpVVodw7WJxWoN5Fs-h8UoqONkKgNo")
    # 改用 2.5 Flash (通常額度較寬裕)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    print("--- 🚀 正在生成 Gemini 2.5 Flash 測試分析 ---")
    response = model.generate_content(prompt)
    print("="*50)
    print(response.text)
    print("="*50)
except Exception as e:
    print(f"❌ 獨立測試失敗: {e}")
