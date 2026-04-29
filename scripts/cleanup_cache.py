"""
🧹 定期快取清理腳本
用於維護快取系統健康
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.cache_manager import get_cache_manager


def main():
    print("=" * 60)
    print(f"🧹 快取清理 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    cache_manager = get_cache_manager()

    # 1. 顯示清理前統計
    print("📊 清理前統計：")
    stats_before = cache_manager.get_stats()
    print(f"  總項目數：{stats_before.get('total_entries', 0)}")
    print(f"  磁碟使用：{stats_before.get('disk_size_mb', 0):.2f} MB")
    print()

    # 2. 清理過期快取
    print("🗑️ 正在清理過期快取...")
    expired_count = cache_manager.cleanup_expired()
    print(f"  ✅ 清理了 {expired_count} 個過期項目")
    print()

    # 3. 顯示清理後統計
    print("📊 清理後統計：")
    stats_after = cache_manager.get_stats()
    print(f"  總項目數：{stats_after.get('total_entries', 0)}")
    print(f"  磁碟使用：{stats_after.get('disk_size_mb', 0):.2f} MB")
    print()

    # 4. 計算節省空間
    saved_mb = stats_before.get("disk_size_mb", 0) - stats_after.get("disk_size_mb", 0)
    saved_items = stats_before.get("total_entries", 0) - stats_after.get(
        "total_entries", 0
    )

    print("💾 清理效果：")
    print(f"  節省項目：{saved_items}")
    print(f"  節省空間：{saved_mb:.2f} MB")
    print()

    # 5. 分類統計
    if "categories" in stats_after:
        print("📁 各類別統計：")
        for category, types in stats_after["categories"].items():
            total = sum(types.values())
            print(f"  • {category}: {total} 項")
            for dtype, count in types.items():
                print(f"      - {dtype}: {count} 項")

    print()
    print("=" * 60)
    print("✅ 清理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
