# 📰 產業新聞查詢功能使用指南

## 📋 功能概述

新增了智能產業新聞查詢功能，可以根據股票所屬產業自動查詢國際新聞，幫助投資者了解產業趨勢。

**核心亮點：**
- ✅ 自動偵測股票產業（半導體、AI、記憶體、光電等）
- ✅ 查詢中英文國際新聞（Google News RSS）
- ✅ 產業情緒分析（正面/負面/中性）
- ✅ 關鍵詞匹配（記憶體缺貨、GPU需求、晶片短缺等）
- ✅ 30分鐘快取機制（避免重複請求）

---

## 🎯 支援產業

### 1. **半導體** 🔵
- **中文關鍵詞**: 半導體、晶圓、晶片、IC設計、封測
- **英文關鍵詞**: semiconductor, chip shortage, GPU demand, AI chip, memory shortage, TSMC, wafer, foundry, ASML, HBM memory, DDR5, chiplet, 3nm process
- **代表股票**: 台積電(2330)、聯發科(2454)、聯電(2303)

### 2. **記憶體** 💾
- **中文關鍵詞**: 記憶體、DRAM、NAND Flash、儲存
- **英文關鍵詞**: memory shortage, DRAM price, NAND flash, HBM, memory demand, storage, SK Hynix, Micron
- **代表股票**: 旺宏(2337)、華邦電(2344)、日月光投控(3711)

### 3. **AI** 🤖
- **中文關鍵詞**: 人工智慧、AI晶片、機器學習、深度學習
- **英文關鍵詞**: AI chip demand, GPU shortage, NVIDIA, AI server, ChatGPT, generative AI, AI accelerator, inference chip
- **代表股票**: 廣達(2382)、華碩(2357)、台郡(6269)

### 4. **光電** 📷
- **中文關鍵詞**: 光學、鏡頭、面板、顯示器
- **英文關鍵詞**: optical lens, camera module, display panel, OLED, mini LED, micro LED, LCD
- **代表股票**: 大立光(3008)、友達(2409)、光寶科(2301)

### 5. **電動車** 🚗
- **中文關鍵詞**: 電動車、電池、充電樁、車用晶片
- **英文關鍵詞**: EV battery, electric vehicle, Tesla, BYD, automotive chip, battery shortage, charging station
- **代表股票**: 台達電(2308)、中興電(1513)、國巨(2327)

### 6. **5G** 📡
- **中文關鍵詞**: 5G、通訊、基地台、射頻
- **英文關鍵詞**: 5G deployment, telecom equipment, RF chip, base station, Qualcomm, MediaTek

### 7. **綠能** 🌱
- **中文關鍵詞**: 太陽能、風電、儲能、電池
- **英文關鍵詞**: solar energy, wind power, energy storage, renewable energy, green energy, battery

### 8. **生技** 💊
- **中文關鍵詞**: 生技、製藥、疫苗、醫療器材
- **英文關鍵詞**: biotech, pharmaceutical, vaccine, medical device, FDA approval, clinical trial

---

## 🚀 使用方式

### **方法1: 使用增強版函數（推薦）**

```python
from utils import get_stock_with_industry_news

# 查詢台積電及其產業新聞
result = get_stock_with_industry_news("2330")

print(f"股票代碼: {result['ticker']}")
print(f"當前價格: ${result['current_price']:.2f}")
print(f"所屬產業: {result['industry']}")
print(f"產業情緒: {result['industry_sentiment']:.1f}")
print(f"產業摘要: {result['industry_summary']}")

# 顯示產業新聞
for news in result['industry_news'][:5]:
    print(f"{news['emoji']} {news['title']}")
    print(f"   分數: {news['score']} | {news['date']}")
```

**輸出範例：**
```
股票代碼: 2330.TW
當前價格: $1780.00
所屬產業: 半導體
產業情緒: 0.5
產業摘要: 半導體產業情緒: 略為正面 (0.5分) | 分析 20 則國際新聞

🔴 AI需求爆發帶動半導體創新高 SIA：2026年全球晶片銷售上看1兆美元
   分數: 3 | 2026-02-07
🟢 開盤／台股開盤重挫逾三百點 半導體信心潰堤盤勢急凍
   分數: -3 | 2026-02-05
```

---

### **方法2: 直接使用 NewsAnalyzer**

```python
from services.news_analyzer import NewsAnalyzer

analyzer = NewsAnalyzer()

# 1. 偵測產業
industry = analyzer.detect_industry("2330", "台積電")
print(f"偵測產業: {industry}")  # 輸出: 半導體

# 2. 查詢產業新聞
en_news = analyzer.get_industry_news("半導體", language="en", max_results=10)
zh_news = analyzer.get_industry_news("半導體", language="zh", max_results=10)

print(f"英文新聞: {len(en_news)} 則")
print(f"中文新聞: {len(zh_news)} 則")

# 3. 分析產業情緒
score, summary, details = analyzer.analyze_industry_sentiment("半導體", detailed=True)

print(f"情緒分數: {score}")
print(f"摘要: {summary}")
print(f"總新聞數: {details['total_news']}")
print(f"中文新聞: {details['zh_news_count']}")
print(f"英文新聞: {details['en_news_count']}")

# 顯示匹配的新聞
for news in details['matched_news'][:10]:
    print(f"{news['emoji']} {news['title']}")
    print(f"  分數: {news['score']} | 關鍵詞: {news['keywords']}")
```

---

## 📊 實際測試結果

### **測試案例：台積電(2330) - 半導體產業**

```
產業情緒: 略為正面 (0.5分)
總新聞數: 20 則（中文: 10, 英文: 10）

熱門新聞：
1. 🔴 AI需求爆發帶動半導體創新高 SIA：2026年全球晶片銷售上看1兆美元 (+3分)
2. 🟢 開盤／台股開盤重挫逾三百點 半導體信心潰堤盤勢急凍 (-3分)
3. ⚪ AI狂潮助攻 全球半導體銷售額有望首度破1兆美元 (0分)
4. 🔴 晶圓代工廠台積電董事長魏哲家與日本首相會面洽談投資事宜 (+1分)
```

### **測試案例：記憶體產業**

```
產業情緒: 略為正面 (0.7分)
總新聞數: 20 則

熱門新聞：
1. 🔴 記憶體價格狂飆！三星、SK海力士、美光出奇招 推「結算後定價」 (+3分)
2. 🔴 中國記憶體也開始飆漲! 華強北一週漲3倍 (+1分)
3. ⚪ 供需緊繃中國兩家記憶體廠傳大幅擴增產能 (0分)
```

---

## 🎨 在 Streamlit UI 中使用

```python
import streamlit as st
from utils import get_stock_with_industry_news

st.title("智能產業分析")

ticker = st.text_input("輸入股票代碼", "2330")

if st.button("查詢產業新聞"):
    with st.spinner("查詢中..."):
        result = get_stock_with_industry_news(ticker)
    
    # 顯示基本資訊
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("股票代碼", result['ticker'])
    with col2:
        st.metric("當前價格", f"${result['current_price']:.2f}")
    with col3:
        st.metric("所屬產業", result['industry'] or "未知")
    
    # 產業情緒
    if result['industry']:
        st.subheader("📊 產業情緒分析")
        sentiment = result['industry_sentiment']
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric("情緒分數", f"{sentiment:.1f}")
        with col_s2:
            st.info(result['industry_summary'])
        
        # 顯示產業新聞
        if result['industry_news']:
            st.subheader("🌍 最新產業新聞")
            
            for idx, news in enumerate(result['industry_news'][:10], 1):
                with st.expander(f"{news['emoji']} {news['title'][:80]}"):
                    st.write(f"**日期**: {news['date']}")
                    st.write(f"**情緒分數**: {news['score']}")
                    if news['keywords']:
                        st.write(f"**關鍵詞**: {', '.join(news['keywords'])}")
```

---

## ⚙️ 技術細節

### **數據來源**
- **Google News RSS Feed** - 即時國際新聞
- **Yahoo 奇摩股市** - 台股個股新聞（既有功能）

### **快取機制**
- 快取時間：30 分鐘
- 快取位置：`data/news_cache/`
- 快取格式：JSON
- 快取鍵值：`industry_{產業}_{語言}`

### **情緒分析邏輯**

**正面關鍵詞（英文產業關鍵詞）：**
- `shortage` (+2) - 缺貨（對供應商有利）
- `demand surge` (+3) - 需求激增
- `strong demand` (+2) - 需求強勁
- `price increase` (+2) - 價格上漲
- `record high` (+3) - 創新高
- `breakthrough` (+2) - 突破

**負面關鍵詞（英文產業關鍵詞）：**
- `oversupply` (-2) - 供過於求
- `declining` (-2) - 下滑
- `slowdown` (-2) - 放緩
- `weak demand` (-2) - 需求疲軟
- `price drop` (-2) - 價格下跌
- `layoff` (-2) - 裁員

**分數計算：**
```
總分數 = Σ(關鍵詞權重)
平均分數 = 總分數 / 新聞數量
```

---

## 🛠️ 自訂產業映射

如需新增產業或調整關鍵詞，編輯 `services/news_analyzer.py`：

```python
self.industry_keywords = {
    "你的產業": {
        "zh": ["中文關鍵詞1", "中文關鍵詞2"],
        "en": ["english keyword 1", "english keyword 2"]
    },
    # 新增更多產業...
}
```

---

## 📈 使用場景

### **1. 投資決策輔助**
- 了解半導體產業是否面臨缺貨（利好）
- 觀察 AI 晶片需求趨勢
- 追蹤記憶體價格變化

### **2. 風險預警**
- 產業情緒驟降（如：-3分以下）
- 負面新聞集中出現（裁員、砍單）
- 供過於求訊號

### **3. 機會挖掘**
- 產業情緒激增（如：+3分以上）
- 需求爆發訊號（shortage, surge）
- 新技術突破（breakthrough, innovation）

### **4. 基本面補充**
- 技術分析 + 產業趨勢
- 個股新聞 + 產業新聞
- 全方位多角度分析

---

## 🔍 故障排除

### **Q1: 查詢不到產業新聞**
**原因**: 網路問題或產業未定義  
**解決**:
1. 檢查網路連接
2. 確認股票產業已定義（見「支援產業」章節）
3. 手動指定產業：`analyzer.get_industry_news("半導體")`

### **Q2: 新聞過時**
**原因**: 快取未過期  
**解決**: 清除快取
```python
analyzer = NewsAnalyzer()
analyzer.clear_cache()
```

### **Q3: 偵測不到產業**
**原因**: 股票未在映射表中  
**解決**: 手動添加到 `detect_industry()` 的 `industry_mapping` 字典

---

## 📝 版本記錄

### **v1.0 (2026-02-08)**
- ✅ 新增 8 大產業支援
- ✅ 中英文新聞查詢
- ✅ 產業情緒分析
- ✅ 自動產業偵測
- ✅快取機制
- ✅ 整合到 `utils.get_stock_with_industry_news()`

---

## 🎓 最佳實踐

1. **定期更新快取** - 每半小時自動更新，保持資訊新鮮
2. **結合個股新聞** - 產業趨勢 + 個股消息 = 完整視角
3. **關注關鍵詞** - "shortage"、"surge"、"record high" 是強訊號
4. **多語言互補** - 中文新聞看台股、英文新聞看國際趨勢
5. **情緒門檻** - 超過 ±2 分時需特別注意

---

**💡 提示**: 此功能最適合用於產業龍頭股票（如台積電、聯發科），因為產業新聞對這些股票影響最直接！
