"""
📊 新聞收集與模型訓練完整演示
展示如何使用新聞情緒特徵進行訓練
"""

import sys
import os

# 將專案根目錄加入 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.weekly_news_collector import WeeklyNewsCollector
from services.ai_service import AIService
from utils import get_stock_data
import pandas as pd

print("=" * 80)
print("📊 新聞收集與訓練整合演示")
print("=" * 80)

# ============================================================================
# 第一部分：新聞收集演示
# ============================================================================
print("\n" + "=" * 80)
print("📰 第一部分：新聞收集與情緒分析")
print("=" * 80)

collector = WeeklyNewsCollector()

# 1. 查看當前周檔案狀態
print("\n1️⃣ 當前周檔案狀態:")
print("-" * 80)
week_data = collector.load_week_data()
print(f"   周期: {week_data['week']}")
print(f"   日期範圍: {week_data['start_date']} ~ {week_data['end_date']}")
print(f"   已收集新聞: {len(week_data.get('news_items', []))} 則")
print(f"   最後更新: {week_data.get('last_updated', 'N/A')}")

# 2. 查看最近新聞摘要
print("\n2️⃣ 最近 7 天新聞摘要:")
print("-" * 80)
summary = collector.get_recent_news_summary(days=7)
print(f"   總新聞數: {summary['total_news']}")
print(f"   平均情緒: {summary['average_sentiment']:.2f}")

if summary["by_category"]:
    print(f"\n   各類別統計:")
    for category, stats in summary["by_category"].items():
        print(
            f"      {category}: {stats['count']} 則, 平均情緒 {stats['avg_sentiment']:.2f}"
        )
else:
    print("   尚無新聞數據")

# 3. 獲取訓練用新聞特徵
print("\n3️⃣ 訓練用新聞特徵:")
print("-" * 80)
features = collector.get_sentiment_features_for_training(lookback_days=7)
for key, value in features.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.2f}")
    else:
        print(f"   {key}: {value}")

# ============================================================================
# 第二部分：AI 模型整合演示
# ============================================================================
print("\n\n" + "=" * 80)
print("🤖 第二部分：AI 模型新聞特徵整合")
print("=" * 80)

service = AIService()

# 1. 檢查模型配置
print("\n1️⃣ AI 模型配置:")
print("-" * 80)
print(f"   模型版本: {service.MODEL_VERSION}")
print(f"   總特徵數: {len(service.feature_columns)}")

# 顯示新聞相關特徵
news_features = [f for f in service.feature_columns if "news" in f.lower()]
print(f"\n   新聞相關特徵 ({len(news_features)} 個):")
for feature in news_features:
    print(f"      - {feature}")

# 2. 演示數據準備流程（不實際訓練）
print("\n2️⃣ 數據準備流程演示:")
print("-" * 80)

try:
    # 獲取台積電數據
    ticker = "2330.TW"
    print(f"\n   正在獲取 {ticker} 的歷史數據...")
    _, df, _, _ = get_stock_data(ticker)

    if not df.empty:
        print(f"   ✅ 成功獲取 {len(df)} 天的歷史數據")

        # 添加市場情緒特徵（包含新聞特徵）
        print(f"\n   正在添加市場情緒與新聞特徵...")
        df_with_features = service._add_market_sentiment_features(df.copy())

        # 檢查新聞特徵是否存在
        print(f"\n   特徵檢查:")
        for feature in news_features:
            if feature in df_with_features.columns:
                value = df_with_features[feature].iloc[-1]
                print(f"      ✅ {feature}: {value}")
            else:
                print(f"      ❌ {feature}: 缺失")

        # 顯示最後幾筆數據的新聞特徵
        if all(f in df_with_features.columns for f in news_features):
            print(f"\n   最近 5 天的新聞特徵值:")
            print("-" * 80)

            # 創建一個小的 DataFrame 顯示
            display_df = df_with_features[news_features].tail(5)
            display_df.index = [
                f"Day {i}" for i in range(len(display_df) - 5, len(display_df))
            ]
            print(display_df.to_string())

    else:
        print("   ⚠️ 無法獲取數據")

except Exception as e:
    print(f"   ❌ 演示失敗: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 第三部分：使用建議
# ============================================================================
print("\n\n" + "=" * 80)
print("💡 第三部分：使用建議")
print("=" * 80)

print(
    """
1. 📰 每日新聞收集：
   - 手動執行：python jobs/daily_news_collector.py collect
   - 自動排程：.\scripts\setup_news_collection_schedule.ps1

2. 🤖 訓練模型（自動使用新聞特徵）：
   from services.ai_service import AIService
   from utils import get_stock_data
   
   service = AIService()
   _, df, _, _ = get_stock_data("2330.TW")
   result, loss = service.train_deep_learning_model("2330.TW", df)

3. 📊 查看新聞狀態：
   - 顯示摘要：python jobs/daily_news_collector.py summary
   - 列出周檔案：python -m services.weekly_news_collector list

4. 🔧 配置 API Key（如果還沒設置）：
   在 .env 檔案中設置：
   SERPAPI_KEY=your_api_key_here

5. 📖 詳細文檔：
   docs/NEWS_COLLECTION_GUIDE.md
"""
)

print("\n" + "=" * 80)
print("✅ 演示完成！")
print("=" * 80)
