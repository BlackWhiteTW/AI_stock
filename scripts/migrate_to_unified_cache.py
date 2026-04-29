"""
🔄 快取遷移工具
將舊有的分散 JSON/CSV 檔案遷移到統一快取系統
"""

import os
import json
import sys
from pathlib import Path
from tqdm import tqdm

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.cache_manager import get_cache_manager


def migrate_news_cache():
    """遷移 news_cache 目錄的 300+ 個 JSON 檔案"""
    print("📦 開始遷移 news_cache...")

    cache_manager = get_cache_manager()
    news_cache_dir = Path("data/news_cache")

    if not news_cache_dir.exists():
        print("⚠️ news_cache 目錄不存在，跳過")
        return 0

    # 獲取所有 JSON 檔案
    json_files = list(news_cache_dir.glob("*.json"))
    migrated = 0

    for json_file in tqdm(json_files, desc="遷移新聞快取"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 從檔案名提取 ticker（例如：2330.TW_hash.json）
            filename = json_file.stem

            # 判斷類型
            if filename.startswith("industry_"):
                category = "industry_news"
                identifier = filename
            elif filename.startswith("serp_"):
                category = "serp_cache"
                identifier = filename.replace("serp_", "")
            else:
                # 股票新聞（ticker_hash.json）
                ticker = filename.split("_")[0] if "_" in filename else filename
                category = "stock_news"
                identifier = ticker

            # 存入統一快取（保留 7 天）
            cache_manager.set(category, identifier, data, ttl=7 * 24 * 3600)
            migrated += 1

        except Exception as e:
            print(f"❌ 遷移失敗 {json_file.name}: {e}")

    print(f"✅ news_cache 遷移完成：{migrated} 個檔案")
    return migrated


def migrate_weekly_news():
    """遷移 weekly_news 目錄"""
    print("📦 開始遷移 weekly_news...")

    cache_manager = get_cache_manager()
    weekly_dir = Path("data/weekly_news")

    if not weekly_dir.exists():
        print("⚠️ weekly_news 目錄不存在，跳過")
        return 0

    json_files = list(weekly_dir.glob("*.json"))
    migrated = 0

    for json_file in tqdm(json_files, desc="遷移週報新聞"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 檔案名格式：2026_W10.json
            week_id = json_file.stem

            # 存入統一快取（保留 90 天）
            cache_manager.set("weekly_news", week_id, data, ttl=90 * 24 * 3600)
            migrated += 1

        except Exception as e:
            print(f"❌ 遷移失敗 {json_file.name}: {e}")

    print(f"✅ weekly_news 遷移完成：{migrated} 個檔案")
    return migrated


def migrate_stock_data_to_parquet():
    """將 stock_data 的 CSV 檔案轉換為 Parquet（更快）"""
    print("📦 開始遷移 stock_data 到 Parquet...")

    import pandas as pd

    cache_manager = get_cache_manager()
    stock_dir = Path("data/stock_data")

    if not stock_dir.exists():
        print("⚠️ stock_data 目錄不存在，跳過")
        return 0

    csv_files = list(stock_dir.glob("*.csv"))
    migrated = 0

    for csv_file in tqdm(csv_files, desc="轉換股票數據"):
        try:
            # 讀取 CSV
            df = pd.read_csv(csv_file, index_col="Date", parse_dates=True)

            if df.empty:
                continue

            # 股票代碼
            ticker = csv_file.stem

            # 存入統一快取（Parquet 格式，保留 30 天）
            cache_manager.set("stock_data", ticker, df, ttl=30 * 24 * 3600)
            migrated += 1

        except Exception as e:
            print(f"❌ 遷移失敗 {csv_file.name}: {e}")

    print(f"✅ stock_data 遷移完成：{migrated} 個檔案")
    return migrated


def cleanup_old_cache():
    """清理舊的快取檔案（可選）"""
    print("\n🧹 清理舊快取檔案...")

    response = input("⚠️ 確定要刪除所有舊的 JSON/CSV 檔案嗎？(yes/no): ")

    if response.lower() != "yes":
        print("⏭️ 跳過清理")
        return

    # 刪除 news_cache 目錄
    news_cache_dir = Path("data/news_cache")
    if news_cache_dir.exists():
        for f in news_cache_dir.glob("*.json"):
            f.unlink()
        print(f"✅ 已清理 news_cache")

    # 不刪除 stock_data（保留作為備份）
    print("💡 stock_data 已保留作為備份")


def main():
    print("=" * 60)
    print("🚀 快取系統遷移工具")
    print("=" * 60)
    print()

    total_migrated = 0

    # 遷移各類快取
    total_migrated += migrate_news_cache()
    total_migrated += migrate_weekly_news()
    total_migrated += migrate_stock_data_to_parquet()

    # 顯示統計
    print("\n" + "=" * 60)
    print("📊 遷移統計")
    print("=" * 60)

    cache_manager = get_cache_manager()
    stats = cache_manager.get_stats()

    print(f"總項目數：{stats['total_entries']}")
    print(f"內存快取：{stats['memory_cache_size']} 項")
    print(f"磁碟使用：{stats['disk_size_mb']:.2f} MB")
    print()
    print("各類別統計：")
    for category, types in stats["categories"].items():
        print(f"  • {category}:")
        for dtype, count in types.items():
            print(f"    - {dtype}: {count} 項")

    # 可選：清理舊檔案
    print()
    cleanup_old_cache()

    print("\n✅ 遷移完成！")
    print(f"📈 總共遷移 {total_migrated} 個檔案")


if __name__ == "__main__":
    main()
