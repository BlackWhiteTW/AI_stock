"""
批量重新訓練模型以整合財務數據
適用於 v5.0 → v5.1 升級

使用方式:
1. 訓練所有模型: python retrain_for_financial.py --all
2. 訓練重要股票: python retrain_for_financial.py --important
3. 訓練指定股票: python retrain_for_financial.py --tickers 2330.TW 0050.TW
"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from services.ai_service import AIService
from utils import get_stock_data
import pandas as pd
import time

def get_important_tickers():
    """重要股票列表（台灣前50大 + 熱門ETF）"""
    return [
        # 台灣權值股
        '2330.TW', '2317.TW', '2454.TW', '2881.TW', '2882.TW',
        '2412.TW', '2308.TW', '2886.TW', '2891.TW', '2303.TW',
        '2002.TW', '1301.TW', '1303.TW', '2884.TW', '2892.TW',
        
        # 熱門 ETF
        '0050.TW', '0056.TW', '006208.TW', '00878.TW',
        '00631L.TW', '00632R.TW',
    ]

def get_all_tickers():
    """獲取所有股票列表"""
    try:
        df = pd.read_csv('data/tw_stock_list.csv')
        return df['Ticker'].tolist()
    except:
        print("⚠️ 無法讀取 tw_stock_list.csv，使用重要股票列表")
        return get_important_tickers()

def retrain_stock(service, ticker, force=True):
    """重新訓練單一股票
    
    Args:
        service: AIService 實例
        ticker: 股票代碼
        force: 是否強制重訓（忽略版本檢查）
    
    Returns:
        (success, message, elapsed_time)
    """
    try:
        # 檢查是否需要重訓
        if not force and ticker in service.models:
            path = os.path.join(service.model_dir, f"{ticker}.pth")
            if os.path.exists(path):
                import torch
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                model_version = checkpoint.get('version', '1.0')
                
                if model_version == service.MODEL_VERSION:
                    return False, f"✓ 已是最新版本 ({model_version})", 0
        
        # 獲取數據
        symbol, hist, current_price, info = get_stock_data(ticker)
        
        if hist.empty or len(hist) < 200:
            return False, f"✗ 數據不足 (僅 {len(hist)} 天)", 0
        
        # 訓練
        print(f"\n{'='*60}")
        print(f"🚀 訓練 {ticker} ({symbol})")
        print(f"{'='*60}")
        
        start_time = time.time()
        result, loss = service.train_deep_learning_model(
            ticker, 
            hist,
            use_transfer_learning=True  # 使用遷移學習加速
        )
        elapsed = time.time() - start_time
        
        print(f"\n結果: {result}")
        
        return True, f"✓ Loss: {loss:.6f}", elapsed
        
    except Exception as e:
        return False, f"✗ 錯誤: {str(e)[:60]}", 0

def main():
    parser = argparse.ArgumentParser(description='重新訓練模型以整合財務數據 (v5.0 → v5.1)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='訓練所有股票')
    group.add_argument('--important', action='store_true', help='僅訓練重要股票')
    group.add_argument('--tickers', nargs='+', help='指定股票代碼（如: 2330.TW 0050.TW）')
    group.add_argument('--outdated', action='store_true', help='僅訓練版本過舊的模型')
    
    parser.add_argument('--max', type=int, help='最多訓練幾支股票')
    parser.add_argument('--skip-existing', action='store_true', help='跳過已訓練的股票')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 批量重新訓練模型 (v5.1 財務數據整合)")
    print("=" * 80)
    
    # 初始化服務
    service = AIService()
    print(f"\n當前模型版本: {service.MODEL_VERSION}")
    print(f"特徵數量: {len(service.feature_columns)} (含 6 個財務特徵)")
    
    # 確定訓練列表
    if args.all:
        ticker_list = get_all_tickers()
        print(f"\n📋 模式: 訓練所有股票")
    elif args.important:
        ticker_list = get_important_tickers()
        print(f"\n📋 模式: 訓練重要股票")
    elif args.outdated:
        outdated = service.get_outdated_models()
        ticker_list = [item['ticker'] for item in outdated]
        print(f"\n📋 模式: 訓練版本過舊的模型")
        print(f"發現 {len(ticker_list)} 個需要更新的模型")
        for item in outdated[:5]:  # 顯示前5個
            print(f"  - {item['ticker']}: {item['old_version']} → {item['current_version']} ({item['reason']})")
        if len(outdated) > 5:
            print(f"  ... 還有 {len(outdated)-5} 個")
    else:
        ticker_list = args.tickers
        print(f"\n📋 模式: 訓練指定股票")
    
    if args.max:
        ticker_list = ticker_list[:args.max]
        print(f"限制最多訓練: {args.max} 支")
    
    print(f"總計: {len(ticker_list)} 支股票\n")
    
    # 訓練統計
    stats = {
        'total': len(ticker_list),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'total_time': 0
    }
    
    # 開始訓練
    for i, ticker in enumerate(ticker_list, 1):
        print(f"\n[{i}/{len(ticker_list)}] {ticker}")
        
        success, message, elapsed = retrain_stock(
            service, 
            ticker, 
            force=not args.skip_existing
        )
        
        if success:
            stats['success'] += 1
            stats['total_time'] += elapsed
            print(f"✅ {message} ({elapsed:.1f}秒)")
        elif "已是最新版本" in message:
            stats['skipped'] += 1
            print(f"⏭️  {message}")
        else:
            stats['failed'] += 1
            print(f"❌ {message}")
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 訓練總結")
    print("=" * 80)
    print(f"總數: {stats['total']}")
    print(f"✅ 成功: {stats['success']}")
    print(f"⏭️  跳過: {stats['skipped']}")
    print(f"❌ 失敗: {stats['failed']}")
    
    if stats['success'] > 0:
        avg_time = stats['total_time'] / stats['success']
        print(f"\n⏱️  平均訓練時間: {avg_time:.1f} 秒/股")
        print(f"⏱️  總訓練時間: {stats['total_time']/60:.1f} 分鐘")
    
    if stats['success'] == stats['total']:
        print("\n🎉 所有模型訓練完成！")
    elif stats['success'] > 0:
        print(f"\n✅ 完成 {stats['success']}/{stats['total']} 支股票的訓練")
    else:
        print("\n⚠️ 未能成功訓練任何模型")

if __name__ == "__main__":
    main()
