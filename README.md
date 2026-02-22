# AI 股票預測與交易系統

> 🤖 使用深度學習 (GRU) 預測台股價格，整合新聞情感分析，提供全方位的投資決策支援

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 核心功能

### 🎯 AI 預測系統
- ✅ **GRU 深度學習模型** - 60天滑動窗口預測
- ✅ **遷移學習加速** - 訓練速度提升 50-70%
- ✅ **高波動股優化** - 自動偵測並調整訓練策略
- ✅ **11項技術指標** - RSI、MACD、KD、ATR等
- ✅ **動態誤差校準** - 基於歷史預測自動修正

### 📰 新聞情感分析 (v2.0)
- ✅ **76個加權關鍵詞** - 三級權重系統 (1, 2, 3)
- ✅ **時間衰減機制** - 新聞越新權重越高
- ✅ **可信度評估** - 0-100% 智慧評分
- ✅ **來源權重** - 財經專業媒體權重更高
- ✅ **快取機制** - 查詢速度提升 25倍
- ✅ **批量分析** - 支援多股票同時分析
- ✅ **市場情緒** - 整體市場氣氛評估

### 📊 視覺化介面
- ✅ **AI 顧問** - 智慧推薦買賣時機
- ✅ **技術分析** - 完整的K線與指標圖表
- ✅ **資產管理** - 模擬帳戶與績效追蹤
- ✅ **歷史記錄** - 預測準確度驗證
- ✅ **對戰模式** - AI vs 人工操作比較

---

## 📋 目錄結構

```
AI_股票/
├── 📁 core/              # 核心交易邏輯
│   ├── trader.py         # 交易執行器
│   └── auto_trader.py    # 自動交易
│
├── 📁 services/          # 服務層
│   ├── ai_service.py     # AI 預測核心 ⭐
│   ├── ai_model_defs.py  # 神經網路架構
│   ├── news_analyzer.py  # 新聞分析 ⭐ v2.0
│   ├── market_data.py    # 市場數據
│   ├── tech_analysis.py  # 技術分析
│   └── stock_strategy.py # 交易策略
│
├── 📁 pages_ui/          # Streamlit UI
│   ├── ui_advisor.py     # AI 顧問頁面
│   ├── ui_analysis.py    # 技術分析頁面
│   ├── ui_asset.py       # 資產管理頁面
│   ├── ui_battle.py      # AI 對戰頁面
│   └── ui_history.py     # 歷史紀錄頁面
│
├── 📁 jobs/              # 排程任務
│   ├── ai_tasks.py       # AI 相關任務
│   └── daily_job.py      # 每日排程
│
├── 📁 data/              # 數據目錄
│   ├── saved_models/     # 已訓練模型 (260支)
│   ├── stock_data/       # 股票歷史數據
│   ├── news_cache/       # 新聞快取
│   └── *.json           # 配置與結果
│
├── 📁 tests/             # 測試腳本
│   └── test_enhanced_news.py
│
├── 📁 scripts/           # 工具腳本
│   ├── batch_train_models.py
│   ├── train_base_model.py
│   └── upgrade_to_news_feature.py
│
├── 📁 docs/              # 文檔
│   ├── README.md         # 主文檔
│   ├── NEWS_UPGRADE_REPORT.md
│   ├── NEWS_QUICK_GUIDE.md
│   └── NEWS_INTEGRATION_GUIDE.md
│
├── app.py               # 主程式入口
├── utils.py             # 工具函數
└── requirements.txt     # 依賴套件
```

---

## 🚀 快速開始

### 1. 環境需求

```bash
Python 3.8+
CUDA 11.0+ (可選，用於 GPU 加速)
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 啟動系統

#### Windows
```bash
啟動模擬系統.bat
```

#### 或手動啟動
```bash
streamlit run app.py
```

### 4. 訪問介面

瀏覽器開啟：`http://localhost:8501`

---

## 💡 使用指南

### AI 預測

```python
from services.ai_service import AIService
from utils import get_stock_data

service = AIService()

# 獲取股票數據
_, hist, _, _ = get_stock_data('2330.TW')

# 預測明日價格（自動包含新聞調整）
pred_price = service.predict_next_price('2330.TW', hist)
print(f"預測價格: {pred_price}")
```

### 新聞分析

```python
from services.news_analyzer import NewsAnalyzer

na = NewsAnalyzer()

# 詳細分析（含可信度）
score, news, details = na.analyze_sentiment('2330', detailed=True)

print(f"情感分數: {score}")
print(f"可信度: {details['confidence']}%")
print(f"情感等級: {details['sentiment_level']}")
```

### 批量分析

```python
# 分析多支股票
stocks = ['2330', '2317', '0050', '2882']
results = na.batch_analyze(stocks, top_n=10)

for r in results:
    print(f"{r['ticker']}: {r['score']:+d} ({r['confidence']}%)")
```

### 市場情緒

```python
# 分析整體市場
market_stocks = ['2330', '2317', '2454', '2882', '2886']
market = na.get_market_sentiment(market_stocks)

print(f"市場分數: {market['market_score']}")
print(f"市場情緒: {market['market_sentiment']}")
```

---

## 📈 模型架構

### GRU 深度學習模型 v3.0

**特徵數量**: 11個
- 價格特徵：Close, Open, High, Low, Volume
- 技術指標：RSI, MACD, K, D, ATR, Volatility

**網路結構**:
```
Input (60 days × 11 features)
    ↓
3-Layer GRU (hidden_dim=128)
    ↓
LayerNorm
    ↓
2-Layer MLP
    ↓
Dropout (0.2)
    ↓
Output (預測價格)
```

**訓練優化**:
- ✅ 遷移學習 - 使用8支大型權值股的預訓練模型
- ✅ 高波動偵測 - CV>60% 啟用特殊訓練
- ✅ Huber Loss - 對異常值更穩健
- ✅ 對數縮放 - 高波動股價格平滑化
- ✅ 動態 Epoch - 普通股100輪，高波動200輪

---

## 🎯 新聞分析系統 v2.0

### 核心改進

| 功能 | 舊版 | 新版 | 提升 |
|------|------|------|------|
| 關鍵詞數 | 24個 | 76個 | 3.2x |
| 權重系統 | 單一 | 三級 | 精準度↑ |
| 時間衰減 | ✗ | ✓ | 相關性↑ |
| 可信度評估 | ✗ | ✓ (0-100%) | 可靠性↑ |
| 來源權重 | ✗ | ✓ (1.0-1.2x) | 準確度↑ |
| 快取機制 | ✗ | ✓ (30分鐘) | 速度25x↑ |

### 情感分數計算

```
最終分數 = Σ(關鍵詞權重 × 來源權重 × 時間權重) / Σ權重
```

### 智慧調整

**最終調整 = 基本調整 × 可信度係數**

| 情感分數 | 基本調整 | 高可信度 | 低可信度 |
|---------|---------|---------|---------|
| ≥5 | ±1.0% | ±1.0% | ±0.4% |
| ≥3 | ±0.7% | ±0.7% | ±0.3% |
| ≥1 | ±0.2% | ±0.2% | ±0.08% |

---

## 📊 效能數據

### 模型訓練

| 指標 | 數值 |
|------|------|
| 已訓練模型 | 50/260支 |
| 訓練成功率 | 100% |
| 平均訓練時間 | 113.7秒/支 |
| 平均 Loss | <0.05 (90%模型) |
| 高波動股比例 | 48% |

### 新聞分析

| 指標 | 數值 |
|------|------|
| 快取加速 | 25x |
| 首次查詢 | ~2.5秒 |
| 快取查詢 | ~0.1秒 |
| 獲取成功率 | >95% |
| 平均新聞數 | 15則/股 |

---

## 🔧 進階功能

### 批量訓練模型

```bash
python scripts/batch_train_models.py
```

### 訓練基礎模型（遷移學習）

```bash
python scripts/train_base_model.py
```

### 升級到新聞特徵版本

```bash
python scripts/upgrade_to_news_feature.py
```

### 完整測試

```bash
python tests/test_enhanced_news.py
```

---

## 📚 文檔

- [新聞升級報告](docs/NEWS_UPGRADE_REPORT.md) - 完整技術說明
- [快速使用指南](docs/NEWS_QUICK_GUIDE.md) - 簡化版說明
- [整合指南](docs/NEWS_INTEGRATION_GUIDE.md) - 實施方案對比

---

## 🛠️ 技術棧

### 核心框架
- **PyTorch** - 深度學習框架
- **Streamlit** - Web UI
- **pandas** - 數據處理
- **yfinance** - 股票數據

### 數據分析
- **scikit-learn** - 機器學習工具
- **ta-lib** - 技術分析
- **BeautifulSoup** - 網頁爬蟲

### 視覺化
- **plotly** - 互動式圖表
- **matplotlib** - 靜態圖表

---

## ⚙️ 配置

### 模型配置 (`services/ai_service.py`)

```python
MODEL_VERSION = "3.0"
lookback = 60  # 回溯天數
feature_columns = 11  # 特徵數量
```

### 新聞配置 (`services/news_analyzer.py`)

```python
cache_duration = 1800  # 30分鐘快取
max_retries = 2  # 最大重試次數
```

---

## 📅 更新日誌

### v3.0 (2026-02-05) - 新聞分析全面升級

#### 新增
- ✅ 新聞可信度評估系統 (0-100%)
- ✅ 76個加權關鍵詞庫
- ✅ 時間衰減機制
- ✅ 來源權重系統
- ✅ 快取機制 (25x加速)
- ✅ 批量分析功能
- ✅ 市場情緒分析

#### 改進
- 🔄 新聞影響根據可信度智慧調整
- 🔄 錯誤處理與重試機制
- 🔄 多種CSS選擇器備援

#### 修復
- 🐛 高波動股訓練不穩定
- 🐛 預測價格base_price計算錯誤
- 🐛 30個模型特徵維度不匹配

### v2.1 (2026-02-04)
- 簡化模型架構
- 增加訓練輪數 (100-200)

### v2.0 (2026-02-03)
- 遷移學習實施
- 特徵優化 (18→11)
- 高波動股優化

---

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

---

## 📄 授權

MIT License

---

## 📞 聯繫

如有問題或建議，請開啟 Issue。

---

**最後更新**: 2026年2月5日  
**版本**: AI Stock System v3.0
