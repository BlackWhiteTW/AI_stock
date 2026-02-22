"""
AutoTrader 參數優化與回測比較工具
=====================================

用途:
1. 自動尋找最佳停損停利參數
2. 比較 v1.0 vs v2.0 績效差異
3. 生成詳細的優化報告
"""

import pandas as pd
import numpy as np
import datetime
import sys
import os

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auto_trader import AutoTrader
from core.auto_trader_v2 import AutoTraderV2
from utils import get_stock_data

def optimize_single_stock(ticker, lookback_days=90):
    """
    對單一股票進行參數優化
    
    Args:
        ticker: 股票代碼 (如 '2330.TW')
        lookback_days: 回測天數
    
    Returns:
        dict: 優化結果
    """
    print(f"\n{'='*60}")
    print(f"開始優化 {ticker}")
    print(f"{'='*60}\n")
    
    # 取得數據
    try:
        name, hist, price, _ = get_stock_data(ticker)
        if hist.empty:
            return {"error": "無數據"}
    except Exception as e:
        return {"error": str(e)}
    
    # 創建 v2.0 交易機器人
    trader = AutoTraderV2(
        account_file='data/ai_account_temp.json',
        initial_balance=1000000
    )
    
    # 定義參數網格
    param_grid = {
        'atr_stop_multiplier': [2.0, 2.5, 3.0],      # 停損倍數
        'atr_profit_multiplier': [3.5, 4.0, 4.5, 5.0], # 停利倍數
        'confidence_threshold': [0.55, 0.60, 0.65]    # 信心度門檻
    }
    
    # 執行優化
    result = trader.optimize_parameters(
        ticker=ticker,
        df=hist,
        param_grid=param_grid,
        lookback_days=lookback_days
    )
    
    return {
        'ticker': ticker,
        'name': name,
        'best_params': result['best_params'],
        'best_score': result['best_score'],
        'all_results': result['all_results'][:5]  # 只保留前 5 名
    }

def compare_v1_vs_v2(ticker, lookback_days=90):
    """
    比較 v1.0 與 v2.0 績效差異
    
    Returns:
        dict: 比較結果
    """
    print(f"\n{'='*60}")
    print(f"v1.0 vs v2.0 績效比較: {ticker}")
    print(f"{'='*60}\n")
    
    # 取得數據
    name, hist, price, _ = get_stock_data(ticker)
    if hist.empty:
        return {"error": "無數據"}
    
    # v1.0 回測
    print("執行 v1.0 回測...")
    trader_v1 = AutoTrader(
        account_file='data/ai_account_v1_temp.json',
        initial_balance=1000000
    )
    msg_v1, df_log_v1 = trader_v1.run_simulation(ticker, hist.copy(), lookback_days=lookback_days)
    
    # v2.0 回測
    print("執行 v2.0 回測...")
    trader_v2 = AutoTraderV2(
        account_file='data/ai_account_v2_temp.json',
        initial_balance=1000000
    )
    msg_v2, df_log_v2 = trader_v2.run_simulation(ticker, hist.copy(), lookback_days=lookback_days)
    
    # 計算績效指標
    def calculate_metrics(df_log, initial_balance):
        if df_log.empty:
            return {}
        
        final_assets = df_log['Assets'].iloc[-1]
        roi = (final_assets - initial_balance) / initial_balance
        
        # 交易次數
        trades = len(df_log[df_log['Action'].isin(['BUY', 'SELL'])])
        
        # 最大回撤
        peak = df_log['Assets'].cummax()
        drawdown = (df_log['Assets'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # 夏普比率
        returns = df_log['Assets'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # 勝率
        buy_actions = df_log[df_log['Action'] == 'BUY']
        sell_actions = df_log[df_log['Action'].isin(['SELL', 'TAKE_PROFIT'])]
        
        if len(buy_actions) > 0 and len(sell_actions) > 0:
            wins = len(df_log[df_log['Action'] == 'TAKE_PROFIT'])
            total_closes = len(sell_actions)
            win_rate = wins / total_closes if total_closes > 0 else 0
        else:
            win_rate = 0
        
        return {
            'final_assets': final_assets,
            'roi': roi,
            'trades': trades,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate
        }
    
    metrics_v1 = calculate_metrics(df_log_v1, 1000000)
    metrics_v2 = calculate_metrics(df_log_v2, 1000000)
    
    # 計算買進持有策略 (Benchmark)
    start_price = hist['Close'].iloc[-lookback_days] if len(hist) > lookback_days else hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    benchmark_roi = (end_price - start_price) / start_price
    
    return {
        'ticker': ticker,
        'name': name,
        'v1_metrics': metrics_v1,
        'v2_metrics': metrics_v2,
        'benchmark_roi': benchmark_roi,
        'improvement': {
            'roi': metrics_v2.get('roi', 0) - metrics_v1.get('roi', 0),
            'sharpe': metrics_v2.get('sharpe_ratio', 0) - metrics_v1.get('sharpe_ratio', 0),
            'win_rate': metrics_v2.get('win_rate', 0) - metrics_v1.get('win_rate', 0)
        }
    }

def batch_optimize_portfolio(tickers, lookback_days=90):
    """
    批量優化多檔股票
    
    Args:
        tickers: 股票代碼列表
        lookback_days: 回測天數
    
    Returns:
        list: 優化結果列表
    """
    results = []
    
    for ticker in tickers:
        try:
            result = optimize_single_stock(ticker, lookback_days)
            results.append(result)
        except Exception as e:
            print(f"優化 {ticker} 失敗: {e}")
            results.append({
                'ticker': ticker,
                'error': str(e)
            })
    
    return results

def generate_optimization_report(results, output_file='optimization_report.txt'):
    """
    生成優化報告
    
    Args:
        results: 優化結果列表
        output_file: 輸出檔案路徑
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AutoTrader v2.0 參數優化報告\n")
        f.write(f"生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            if 'error' in result:
                f.write(f"\n{result['ticker']}: 錯誤 - {result['error']}\n")
                continue
            
            f.write(f"\n股票: {result['name']} ({result['ticker']})\n")
            f.write("-"*80 + "\n")
            
            f.write(f"最佳參數:\n")
            for param, value in result['best_params'].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write(f"\n最佳評分: {result['best_score']:.4f}\n")
            
            f.write(f"\n前 5 名參數組合:\n")
            for idx, r in enumerate(result['all_results'][:5], 1):
                f.write(f"\n  {idx}. 參數: {r['params']}\n")
                f.write(f"     ROI: {r['roi']:.2%} | Sharpe: {r['sharpe']:.2f} | 評分: {r['score']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ 報告已保存至: {output_file}")

def generate_comparison_report(comparison_results, output_file='v1_vs_v2_comparison.txt'):
    """
    生成 v1 vs v2 比較報告
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AutoTrader v1.0 vs v2.0 績效比較報告\n")
        f.write(f"生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for result in comparison_results:
            if 'error' in result:
                f.write(f"\n{result['ticker']}: 錯誤\n")
                continue
            
            f.write(f"\n股票: {result['name']} ({result['ticker']})\n")
            f.write("-"*80 + "\n")
            
            f.write(f"\nBenchmark (買進持有):\n")
            f.write(f"  ROI: {result['benchmark_roi']:.2%}\n")
            
            f.write(f"\nv1.0 績效:\n")
            v1 = result['v1_metrics']
            f.write(f"  最終資產: ${v1.get('final_assets', 0):,.0f}\n")
            f.write(f"  ROI: {v1.get('roi', 0):.2%}\n")
            f.write(f"  交易次數: {v1.get('trades', 0)}\n")
            f.write(f"  最大回撤: {v1.get('max_drawdown', 0):.2%}\n")
            f.write(f"  夏普比率: {v1.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"  勝率: {v1.get('win_rate', 0):.2%}\n")
            
            f.write(f"\nv2.0 績效:\n")
            v2 = result['v2_metrics']
            f.write(f"  最終資產: ${v2.get('final_assets', 0):,.0f}\n")
            f.write(f"  ROI: {v2.get('roi', 0):.2%}\n")
            f.write(f"  交易次數: {v2.get('trades', 0)}\n")
            f.write(f"  最大回撤: {v2.get('max_drawdown', 0):.2%}\n")
            f.write(f"  夏普比率: {v2.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"  勝率: {v2.get('win_rate', 0):.2%}\n")
            
            f.write(f"\n改進幅度:\n")
            imp = result['improvement']
            f.write(f"  ROI 提升: {imp['roi']:+.2%}\n")
            f.write(f"  夏普比率提升: {imp['sharpe']:+.2f}\n")
            f.write(f"  勝率提升: {imp['win_rate']:+.2%}\n")
            
            # 判定結論
            if imp['roi'] > 0 and imp['sharpe'] > 0:
                f.write(f"\n✅ 結論: v2.0 明顯優於 v1.0\n")
            elif imp['roi'] > 0:
                f.write(f"\n⚠️ 結論: v2.0 ROI 較高,但風險調整後表現相近\n")
            else:
                f.write(f"\n❌ 結論: v2.0 在此股票表現不如 v1.0\n")
            
            f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ 比較報告已保存至: {output_file}")

# CLI 介面
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoTrader 優化與比較工具')
    parser.add_argument('--mode', choices=['optimize', 'compare', 'batch'],
                       default='compare', help='執行模式')
    parser.add_argument('--ticker', type=str, default='0050.TW',
                       help='股票代碼 (如 2330.TW)')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='批量優化的股票列表')
    parser.add_argument('--days', type=int, default=90,
                       help='回測天數')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        # 單一股票優化
        result = optimize_single_stock(args.ticker, args.days)
        generate_optimization_report([result], f'optimize_{args.ticker.replace(".", "_")}.txt')
    
    elif args.mode == 'compare':
        # v1 vs v2 比較
        result = compare_v1_vs_v2(args.ticker, args.days)
        generate_comparison_report([result], f'compare_{args.ticker.replace(".", "_")}.txt')
    
    elif args.mode == 'batch':
        # 批量優化
        if not args.tickers:
            print("錯誤: 請使用 --tickers 指定股票列表")
            print("例: python optimize_trader.py --mode batch --tickers 2330.TW 0050.TW 2317.TW")
        else:
            results = batch_optimize_portfolio(args.tickers, args.days)
            generate_optimization_report(results, f'batch_optimize_{datetime.datetime.now().strftime("%Y%m%d")}.txt')
    
    print("\n✅ 全部完成!")
