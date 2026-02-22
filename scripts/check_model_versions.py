"""
模型版本升級與兼容性說明
"""

import os
import glob

def check_model_versions():
    """檢查所有模型的版本狀態"""
    model_dir = "data/saved_models"
    
    if not os.path.exists(model_dir):
        print("模型目錄不存在")
        return
    
    print("\n" + "="*60)
    print("模型版本檢查報告")
    print("="*60)
    
    import torch
    
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    version_stats = {}
    outdated_models = []
    
    for model_path in model_files:
        ticker = os.path.basename(model_path).replace('.pth', '')
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            version = checkpoint.get('version', '1.0')
            
            if version not in version_stats:
                version_stats[version] = []
            version_stats[version].append(ticker)
            
            if version != '5.0':  # 當前版本
                outdated_models.append((ticker, version))
                
        except Exception as e:
            print(f"❌ {ticker}: 讀取失敗 - {e}")
    
    print(f"\n📊 版本分布:")
    for version, tickers in sorted(version_stats.items()):
        print(f"  v{version}: {len(tickers)} 個模型")
    
    if outdated_models:
        print(f"\n⚠️  需要更新的模型 ({len(outdated_models)} 個):")
        for ticker, version in outdated_models[:10]:  # 只顯示前10個
            print(f"  • {ticker} (v{version} → v5.0)")
        
        if len(outdated_models) > 10:
            print(f"  ... 還有 {len(outdated_models) - 10} 個")
        
        print(f"\n💡 建議:")
        print(f"  1. 使用對戰頁面的「批量訓練熱門股」更新重要股票")
        print(f"  2. 或執行: python scripts/batch_retrain_models.py")
    else:
        print(f"\n✅ 所有模型都是最新版本 (v5.0)")
    
    print("="*60 + "\n")

def batch_retrain_outdated():
    """批量重新訓練過時的模型"""
    from ai_advisor import AIAdvisor
    from utils import get_stock_data
    
    advisor = AIAdvisor()
    model_dir = "data/saved_models"
    
    import torch
    
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    outdated = []
    
    # 找出過時模型
    for model_path in model_files:
        ticker = os.path.basename(model_path).replace('.pth', '')
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            version = checkpoint.get('version', '1.0')
            if version != '5.0':
                outdated.append(ticker)
        except:
            pass
    
    if not outdated:
        print("✅ 沒有需要更新的模型")
        return
    
    print(f"\n開始批量訓練 {len(outdated)} 個過時模型...")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(outdated, 1):
        print(f"\n[{i}/{len(outdated)}] 訓練 {ticker}...")
        
        try:
            # 獲取數據
            full_ticker, hist, _, _ = get_stock_data(ticker)
            
            if hist.empty:
                print(f"  ❌ 無法獲取數據")
                fail_count += 1
                continue
            
            # 訓練
            result, _ = advisor.train_deep_learning_model(full_ticker, hist)
            
            if "成功" in result or "完成" in result:
                print(f"  ✅ 訓練成功")
                success_count += 1
            else:
                print(f"  ⚠️  訓練異常: {result}")
                fail_count += 1
                
        except Exception as e:
            print(f"  ❌ 訓練失敗: {e}")
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"批量訓練完成:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失敗: {fail_count}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        batch_retrain_outdated()
    else:
        check_model_versions()
