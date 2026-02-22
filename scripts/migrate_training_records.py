"""
訓練記錄遷移工具
將舊的 training_history.json 轉換為新的獨立文件存儲格式
"""
import os
import json
from datetime import datetime


def migrate_training_records():
    """遷移訓練記錄到新格式"""
    
    old_file = "data/training_history.json"
    new_dir = "data/training_records"
    
    print("🔄 開始遷移訓練記錄...")
    
    # 檢查舊文件是否存在
    if not os.path.exists(old_file):
        print(f"❌ 找不到舊記錄文件: {old_file}")
        print("ℹ️  如果這是新安裝，則無需遷移")
        return
    
    # 創建新文件夾
    os.makedirs(new_dir, exist_ok=True)
    
    # 讀取舊記錄
    try:
        with open(old_file, 'r', encoding='utf-8') as f:
            old_records = json.load(f)
        
        print(f"📊 找到 {len(old_records)} 條舊記錄")
        
        # 轉換每條記錄
        migrated_count = 0
        date_counters = {}  # 記錄每個日期的計數器
        
        for idx, record in enumerate(old_records):
            try:
                # 獲取時間戳
                timestamp = record.get('timestamp', '')
                if timestamp:
                    # 從 ISO 格式提取日期
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime('%Y%m%d')
                else:
                    # 沒有時間戳，使用當前日期
                    date_str = datetime.now().strftime('%Y%m%d')
                
                # 同一天多次訓練自動編號
                if date_str not in date_counters:
                    date_counters[date_str] = 1
                    filename = f"training_{date_str}.json"
                else:
                    date_counters[date_str] += 1
                    filename = f"training_{date_str}_{date_counters[date_str]:02d}.json"
                
                filepath = os.path.join(new_dir, filename)
                
                # 保存獨立文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                
                migrated_count += 1
                
                if (idx + 1) % 10 == 0:
                    print(f"  • 已遷移 {idx + 1}/{len(old_records)} 條記錄...")
                    
            except Exception as e:
                print(f"  ⚠️  記錄 {idx} 遷移失敗: {e}")
                continue
        
        print(f"\n✅ 成功遷移 {migrated_count}/{len(old_records)} 條記錄")
        
        # 詢問是否刪除舊文件
        print(f"\n舊文件位置: {old_file}")
        response = input("是否刪除舊文件？(y/n): ").strip().lower()
        
        if response == 'y':
            # 備份舊文件
            backup_file = old_file.replace('.json', '_backup.json')
            os.rename(old_file, backup_file)
            print(f"✅ 舊文件已備份至: {backup_file}")
        else:
            print("ℹ️  舊文件已保留")
        
        print(f"\n🎉 遷移完成！新記錄位於: {new_dir}")
        
    except Exception as e:
        print(f"❌ 遷移失敗: {e}")
        import traceback
        traceback.print_exc()


def verify_migration():
    """驗證遷移結果"""
    
    new_dir = "data/training_records"
    
    print("\n🔍 驗證遷移結果...")
    
    if not os.path.exists(new_dir):
        print(f"❌ 新文件夾不存在: {new_dir}")
        return
    
    files = [f for f in os.listdir(new_dir) if f.endswith('.json')]
    
    print(f"📊 統計信息:")
    print(f"  • 記錄文件數: {len(files)}")
    
    total_size = 0
    valid_count = 0
    
    for filename in files:
        filepath = os.path.join(new_dir, filename)
        file_size = os.path.getsize(filepath)
        total_size += file_size
        
        # 驗證 JSON 格式
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                valid_count += 1
        except:
            print(f"  ⚠️  無效文件: {filename}")
    
    print(f"  • 有效記錄: {valid_count}/{len(files)}")
    print(f"  • 總磁盤使用: {total_size / 1024:.2f} KB")
    print(f"  • 平均文件大小: {total_size / len(files) / 1024:.2f} KB" if len(files) > 0 else "  • 平均文件大小: 0 KB")
    
    # 顯示幾個示例
    if files:
        print(f"\n📋 示例文件:")
        for filename in sorted(files)[:3]:
            print(f"  • {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("📦 訓練記錄遷移工具")
    print("=" * 60)
    print()
    
    migrate_training_records()
    verify_migration()
    
    print("\n" + "=" * 60)
    print("遷移完成！您可以在 Tab 7 的「訓練歷史與統計」查看記錄")
    print("=" * 60)
