
import sys
import os
import pandas as pd
import time

# 將專案根目錄加入路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.market_data import MarketDataManager

def test_consolidation():
    print("🚀 開始測試 CSV 整合至 Parquet 功能...")
    
    manager = MarketDataManager()
    output_path = 'data/market_snapshot.parquet'
    
    # 1. 執行整合
    start_time = time.time()
    result_path = manager.consolidate_to_parquet(output_path)
    elapsed = time.time() - start_time
    
    if not result_path or not os.path.exists(result_path):
        print("❌ 整合失敗：找不到輸出的 Parquet 檔案。")
        return
    
    print(f"✅ 整合完成！耗時: {elapsed:.2f} 秒")
    
    # 2. 驗證數據
    try:
        df = pd.read_parquet(result_path)
        print("\n📊 --- 數據驗證結果 ---")
        print(f"總行數: {len(df):,}")
        print(f"包含股票數: {df['Ticker'].nunique()}")
        print(f"檔案大小: {os.path.getsize(result_path) / 1024 / 1024:.2f} MB")
        
        print("\n📝 數據樣本 (前 5 筆):")
        print(df.head())
        
        # 測試讀取速度
        start_read = time.time()
        _ = pd.read_parquet(result_path)
        read_elapsed = time.time() - start_read
        print(f"\n⚡ Parquet 完整讀取耗時: {read_elapsed:.4f} 秒")
        
    except Exception as e:
        print(f"❌ 數據驗證出錯: {e}")

if __name__ == "__main__":
    test_consolidation()
