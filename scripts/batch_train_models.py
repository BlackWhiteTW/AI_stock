"""
批量訓練AI模型
訓練50支股票，每個訓練之間有休息間隔
"""
import time
import pandas as pd
from services.ai_service import AIService
from utils import get_stock_data
from datetime import datetime

def main():
    # 讀取股票列表
    df = pd.read_csv('tw_stock_list.csv')
    tickers = df['Ticker'].tolist()[:50]  # 取前50支
    
    print("=" * 70)
    print("🚀 批量訓練AI模型 (v3.0 遷移學習版本)")
    print("=" * 70)
    print(f"訓練數量: {len(tickers)} 支股票")
    print(f"休息間隔: 10秒")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    service = AIService()
    
    success_count = 0
    fail_count = 0
    total_time = 0
    
    for idx, ticker in enumerate(tickers, 1):
        stock_name = df[df['Ticker'] == ticker]['名稱'].values[0] if len(df[df['Ticker'] == ticker]) > 0 else ticker
        
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(tickers)}] 訓練: {ticker} ({stock_name})")
        print(f"{'='*70}")
        
        try:
            # 獲取股票數據
            _, hist, _, _ = get_stock_data(ticker)
            
            if hist.empty or len(hist) < 200:
                print(f"⚠️  跳過 {ticker}：數據不足 (僅{len(hist)}天)")
                fail_count += 1
                continue
            
            # 訓練模型
            start_time = time.time()
            
            def progress_callback(current, total, loss, elapsed):
                """訓練進度回調"""
                progress = (current / total) * 100
                print(f"  訓練進度: {current}/{total} ({progress:.0f}%) | Loss: {loss:.6f} | 耗時: {elapsed:.1f}s", end='\r')
            
            result, final_loss = service.train_deep_learning_model(
                ticker, 
                hist, 
                progress_callback=progress_callback
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"\n✅ {ticker} 訓練完成！")
            print(f"   最終Loss: {final_loss:.6f}")
            print(f"   耗時: {elapsed:.1f}秒")
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ {ticker} 訓練失敗: {str(e)}")
            fail_count += 1
        
        # 統計資訊
        avg_time = total_time / success_count if success_count > 0 else 0
        remaining = (len(tickers) - idx) * avg_time if avg_time > 0 else 0
        
        print(f"\n📊 進度統計:")
        print(f"   成功: {success_count} | 失敗: {fail_count} | 剩餘: {len(tickers) - idx}")
        print(f"   平均耗時: {avg_time:.1f}秒/支 | 預估剩餘時間: {remaining/60:.1f}分鐘")
        
        # 每個股票訓練後休息10秒（最後一個不休息）
        if idx < len(tickers):
            print(f"\n⏸️  休息10秒，讓GPU降溫...")
            for i in range(10, 0, -1):
                print(f"   {i}秒...", end='\r')
                time.sleep(1)
            print("   繼續訓練...      ")
    
    # 最終統計
    print("\n" + "=" * 70)
    print("🎉 批量訓練完成！")
    print("=" * 70)
    print(f"總訓練數: {len(tickers)} 支")
    print(f"成功: {success_count} 支 ({success_count/len(tickers)*100:.1f}%)")
    print(f"失敗: {fail_count} 支 ({fail_count/len(tickers)*100:.1f}%)")
    print(f"總耗時: {total_time/60:.1f} 分鐘")
    print(f"平均速度: {total_time/success_count:.1f} 秒/支" if success_count > 0 else "")
    print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
