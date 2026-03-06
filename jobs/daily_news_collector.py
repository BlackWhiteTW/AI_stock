"""
🤖 每日新聞自動收集任務
在每天早上自動收集市場新聞並存入周檔案
支援手動執行或整合到排程任務中
"""

import sys
import os

# 將專案根目錄加入 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.weekly_news_collector import WeeklyNewsCollector
from datetime import datetime
import traceback


def run_daily_news_collection():
    """執行每日新聞收集"""
    print("=" * 80)
    print(f"🤖 每日新聞自動收集任務")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # 初始化新聞收集器
        collector = WeeklyNewsCollector()

        # 執行每日收集
        stats = collector.collect_daily_news()

        # 顯示收集結果
        print("\n✅ 新聞收集完成!")
        print(f"📊 統計資訊:")
        print(f"   - 當前周檔案: {stats['week']}")
        print(f"   - 本次收集: {stats['news_collected_today']} 則新聞")
        print(f"   - 本周累計: {stats['total_news_this_week']} 則新聞")
        print(f"   - 更新時間: {stats['last_updated']}")

        # 清理舊檔案 (保留 12 周)
        print("\n🧹 檢查是否需要清理舊檔案...")
        collector.cleanup_old_files(keep_weeks=12)

        return True

    except Exception as e:
        print(f"\n❌ 新聞收集失敗: {e}")
        print(traceback.format_exc())
        return False


def show_recent_summary():
    """顯示最近新聞摘要"""
    try:
        collector = WeeklyNewsCollector()
        summary = collector.get_recent_news_summary(days=7)

        print("\n" + "=" * 80)
        print("📊 最近 7 天新聞摘要")
        print("=" * 80)
        print(f"總新聞數: {summary['total_news']}")
        print(f"平均情緒: {summary['average_sentiment']:.2f}")

        if summary["by_category"]:
            print(f"\n各類別統計:")
            for category, stats in summary["by_category"].items():
                print(
                    f"  {category}: {stats['count']} 則, 平均情緒 {stats['avg_sentiment']:.2f}"
                )

        print(f"\n最後更新: {summary['latest_update']}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 讀取摘要失敗: {e}")


def main():
    """主函數"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "collect":
            # 執行收集
            success = run_daily_news_collection()
            sys.exit(0 if success else 1)

        elif command == "summary":
            # 顯示摘要
            show_recent_summary()

        elif command == "help":
            print("可用命令:")
            print("  collect  - 執行每日新聞收集")
            print("  summary  - 顯示最近7天新聞摘要")
            print("  help     - 顯示此幫助訊息")

        else:
            print(f"未知命令: {command}")
            print("使用 'help' 查看可用命令")

    else:
        # 預設執行收集
        run_daily_news_collection()


if __name__ == "__main__":
    main()
