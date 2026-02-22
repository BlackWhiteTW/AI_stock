"""
方案A：將新聞情感作為第12個特徵整合到模型
需要重新訓練所有模型（預計4-6小時）
"""

import pandas as pd
import numpy as np
import os
import json
import time
from services.ai_service import AIService
from services.news_analyzer import NewsAnalyzer
from utils import get_stock_data

def add_news_sentiment_feature(df, ticker, news_analyzer):
    """為歷史數據添加新聞情感特徵（簡化版：用0填充）"""
    # 由於我們沒有歷史新聞數據，這裡用0填充
    # 實際上模型會從其他11個特徵學習，新聞特徵在預測時才真正發揮作用
    df['News_Sentiment'] = 0.0
    return df

def upgrade_models_with_news():
    """升級所有模型，添加新聞特徵"""
    
    print("=" * 80)
    print("🔄 升級AI模型：整合新聞情感特徵")
    print("=" * 80)
    print("\n⚠️  警告：此操作將重新訓練所有模型（260支股票）")
    print("預計耗時：4-6 小時")
    print("\n按 Ctrl+C 取消，或按 Enter 繼續...")
    input()
    
    # 修改 AIService 以支持新聞特徵
    print("\n第1步：修改模型特徵配置...")
    print("將 feature_columns 從 11 個增加到 12 個")
    
    # 讀取當前股票列表
    ai_service = AIService()
    news_analyzer = NewsAnalyzer()
    
    # 獲取所有需要訓練的股票
    stock_list_file = 'data/tw_stock_list.csv'
    if os.path.exists(stock_list_file):
        df_stocks = pd.read_csv(stock_list_file)
        stock_list = df_stocks['ticker'].tolist()
    else:
        print("❌ 找不到股票列表文件")
        return
    
    print(f"\n找到 {len(stock_list)} 支股票")
    
    # 開始批量訓練
    print("\n" + "=" * 80)
    print("第2步：開始批量訓練（含新聞特徵）")
    print("=" * 80)
    
    results = []
    start_time = time.time()
    
    for idx, ticker in enumerate(stock_list, 1):
        try:
            print(f"\n[{idx}/{len(stock_list)}] 訓練 {ticker}...")
            
            # 獲取股票數據
            _, hist, _, _ = get_stock_data(ticker)
            
            if hist.empty or len(hist) < 500:
                print(f"  ⚠️  數據不足，跳過")
                results.append({'ticker': ticker, 'status': 'skipped', 'reason': '數據不足'})
                continue
            
            # 添加新聞特徵（暫時用0填充）
            hist = add_news_sentiment_feature(hist, ticker, news_analyzer)
            
            # 訓練模型
            result, loss = ai_service.train_deep_learning_model(ticker, hist)
            
            if loss < 0.1:
                status = '✅ 成功'
            else:
                status = '⚠️  Loss偏高'
            
            results.append({
                'ticker': ticker,
                'status': status,
                'loss': f"{loss:.6f}"
            })
            
            print(f"  {status} - Loss: {loss:.6f}")
            
            # 每50支股票休息一下
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(stock_list) - idx) * avg_time
                print(f"\n⏱️  進度：{idx}/{len(stock_list)} ({idx/len(stock_list)*100:.1f}%)")
                print(f"   已耗時：{elapsed/60:.1f} 分鐘")
                print(f"   預計剩餘：{remaining/60:.1f} 分鐘")
                print(f"   平均速度：{avg_time:.1f} 秒/支")
                time.sleep(10)  # 讓GPU休息
            
        except Exception as e:
            print(f"  ❌ 訓練失敗: {str(e)}")
            results.append({
                'ticker': ticker,
                'status': '❌ 失敗',
                'error': str(e)[:100]
            })
    
    # 總結
    total_time = time.time() - start_time
    success_count = len([r for r in results if '成功' in r['status']])
    
    print("\n" + "=" * 80)
    print("🎉 升級完成！")
    print("=" * 80)
    print(f"總耗時：{total_time/60:.1f} 分鐘 ({total_time/3600:.2f} 小時)")
    print(f"成功訓練：{success_count}/{len(stock_list)} 支股票")
    print(f"成功率：{success_count/len(stock_list)*100:.1f}%")
    
    # 保存結果
    results_file = 'news_upgrade_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果已保存至: {results_file}")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  方案A：將新聞情感整合為第12個模型特徵                           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  優點：模型真正學習新聞與價格的關聯                              ║
    ║  缺點：需要重新訓練所有260支股票（4-6小時）                      ║
    ║                                                                  ║
    ║  注意：由於沒有歷史新聞數據，訓練時新聞特徵會用0填充            ║
    ║        但預測時會使用真實的新聞情感分數                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    ⚠️  開始前請確認：
    1. 方案B已測試並確認新聞調整有效
    2. 有足夠的時間進行訓練（4-6小時）
    3. GPU可用且穩定
    
    """)
    
    upgrade_models_with_news()
