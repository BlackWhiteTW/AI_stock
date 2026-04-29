"""
📊 快速查看快取狀態
"""

import os
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.cache_manager import get_cache_manager


def format_bytes(bytes):
    """格式化位元組"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def main():
    print("=" * 70)
    print("📊 快取系統狀態")
    print("=" * 70)
    print()

    cache_manager = get_cache_manager()
    stats = cache_manager.get_stats()

    # 總覽
    print("🔍 總覽")
    print("-" * 70)
    print(f"總項目數        : {stats.get('total_entries', 0):,}")
    print(f"內存快取大小    : {stats.get('memory_cache_size', 0):,} 項")
    print(f"磁碟使用        : {stats.get('disk_size_mb', 0):.2f} MB")
    print()

    # 各類別詳情
    if "categories" in stats and stats["categories"]:
        print("📁 各類別詳情")
        print("-" * 70)

        for category, types in sorted(stats["categories"].items()):
            total = sum(types.values())
            print(f"\n{category.upper()}")
            print(f"  總計: {total:,} 項")

            for dtype, count in types.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"    • {dtype:12s}: {count:6,} 項 ({percentage:5.1f}%)")
    else:
        print("⚠️ 暫無快取資料")

    print()

    # 快取目錄資訊
    print("📂 快取目錄")
    print("-" * 70)

    cache_dir = Path("data/unified_cache")
    if cache_dir.exists():
        # SQLite 資料庫大小
        db_path = cache_dir / "cache_index.db"
        if db_path.exists():
            db_size = db_path.stat().st_size
            print(f"SQLite 資料庫   : {format_bytes(db_size)}")

        # Parquet 目錄大小
        parquet_dir = cache_dir / "parquet"
        if parquet_dir.exists():
            parquet_size = sum(
                f.stat().st_size for f in parquet_dir.rglob("*") if f.is_file()
            )
            parquet_count = len(list(parquet_dir.rglob("*.parquet")))
            print(
                f"Parquet 檔案    : {parquet_count:,} 個，{format_bytes(parquet_size)}"
            )
    else:
        print("⚠️ 快取目錄不存在")

    print()

    # 舊快取提示
    old_news_cache = Path("data/news_cache")
    if old_news_cache.exists():
        old_files = list(old_news_cache.glob("*.json"))
        if old_files:
            print("⚠️ 發現舊快取")
            print("-" * 70)
            print(f"data/news_cache/ 仍有 {len(old_files)} 個舊檔案")
            print("建議執行遷移：python scripts\\migrate_to_unified_cache.py")
            print()

    print("=" * 70)
    print("💡 提示")
    print("-" * 70)
    print("• 清理過期快取: python scripts\\cleanup_cache.py")
    print("• 執行效能測試: python tests\\test_cache_performance.py")
    print("• 遷移舊快取  : python scripts\\migrate_to_unified_cache.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
