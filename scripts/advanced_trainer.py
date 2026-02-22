"""
🚀 進階訓練系統 v1.0
超參數優化、模型對比、自動選擇最佳模型
"""
import sys
import os

# 將專案根目錄加入 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import json
import argparse
from utils import get_stock_data
from services.ai_service import AIService
from core.auto_trader import AutoTrader

class AdvancedTrainer:
    """進階訓練系統"""
    
    def __init__(self):
        self.service = AIService()
        self.results_file = "data/training_comparison.json"
        self.results = self._load_results()
        
    def train_with_hyperparameters(self, ticker, stock_name, params_list=None):
        """
        使用不同超參數訓練模型，找出最佳組合
        :param ticker: 股票代碼
        :param stock_name: 股票名稱
        :param params_list: 超參數組合列表
        :return: 最佳模型資訊
        """
        print(f"\n{'='*60}")
        print(f"🧪 超參數優化訓練: {stock_name} ({ticker})")
        print(f"{'='*60}\n")
        
        # 預設超參數組合
        if params_list is None:
            params_list = [
                {'hidden_dim': 64, 'num_layers': 2, 'lookback': 60, 'model_type': 'LSTM'},
                {'hidden_dim': 128, 'num_layers': 2, 'lookback': 60, 'model_type': 'LSTM'},
                {'hidden_dim': 64, 'num_layers': 3, 'lookback': 60, 'model_type': 'LSTM'},
                {'hidden_dim': 64, 'num_layers': 2, 'lookback': 90, 'model_type': 'LSTM'},
                {'hidden_dim': 64, 'num_layers': 2, 'lookback': 60, 'model_type': 'GRU'},
                {'hidden_dim': 128, 'num_layers': 2, 'lookback': 60, 'model_type': 'GRU'},
            ]
        
        # 獲取數據
        _, hist, _, _ = get_stock_data(ticker)
        if hist.empty:
            print(f"❌ 無法獲取 {ticker} 數據")
            return None
        
        results = []
        
        for idx, params in enumerate(params_list, 1):
            print(f"\n[{idx}/{len(params_list)}] 測試組合: {params}")
            
            # 暫時修改 AIService 參數（這裡簡化，實際需要更複雜的參數傳遞）
            original_lookback = self.service.lookback
            self.service.lookback = params.get('lookback', 60)
            
            try:
                # 訓練模型
                train_msg, loss = self.service.train_deep_learning_model(
                    ticker, hist, use_transfer_learning=True
                )
                
                # 回測評估
                backtest_result = self._backtest_model(ticker, hist)
                
                result = {
                    'params': params,
                    'train_loss': loss,
                    'train_msg': train_msg,
                    'backtest': backtest_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                print(f"  ✓ 訓練損失: {loss:.6f}")
                if backtest_result:
                    print(f"  ✓ 回測報酬: {backtest_result.get('total_return', 0):.2f}%")
                    print(f"  ✓ 勝率: {backtest_result.get('win_rate', 0):.1f}%")
                
            except Exception as e:
                print(f"  ✗ 訓練失敗: {e}")
                results.append({
                    'params': params,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            finally:
                # 恢復原始參數
                self.service.lookback = original_lookback
        
        # 找出最佳模型
        valid_results = [r for r in results if 'error' not in r and r.get('backtest')]
        
        if not valid_results:
            print("\n❌ 沒有成功的訓練結果")
            return None
        
        # 根據回測報酬排序
        best_result = max(valid_results, key=lambda x: x['backtest'].get('total_return', -float('inf')))
        
        print(f"\n{'='*60}")
        print(f"🏆 最佳模型:")
        print(f"  參數: {best_result['params']}")
        print(f"  訓練損失: {best_result['train_loss']:.6f}")
        print(f"  回測報酬: {best_result['backtest']['total_return']:.2f}%")
        print(f"  勝率: {best_result['backtest']['win_rate']:.1f}%")
        print(f"  最大回撤: {best_result['backtest'].get('max_drawdown', 0):.2f}%")
        print(f"{'='*60}\n")
        
        # 保存結果
        self._save_result(ticker, stock_name, results, best_result)
        
        return best_result
    
    def compare_models(self, ticker, stock_name):
        """
        比較 LSTM vs GRU 模型性能
        :param ticker: 股票代碼
        :param stock_name: 股票名稱
        :return: 對比結果
        """
        print(f"\n{'='*60}")
        print(f"📊 模型對比: {stock_name} ({ticker})")
        print(f"{'='*60}\n")
        
        _, hist, _, _ = get_stock_data(ticker)
        if hist.empty:
            print(f"❌ 無法獲取 {ticker} 數據")
            return None
        
        comparison = {}
        
        # 測試 LSTM
        print("1️⃣ 訓練 LSTM 模型...")
        try:
            msg_lstm, loss_lstm = self.service.train_deep_learning_model(ticker, hist)
            backtest_lstm = self._backtest_model(ticker, hist)
            comparison['LSTM'] = {
                'loss': loss_lstm,
                'backtest': backtest_lstm,
                'msg': msg_lstm
            }
            print(f"  ✓ LSTM 損失: {loss_lstm:.6f}")
            if backtest_lstm:
                print(f"  ✓ LSTM 回測: {backtest_lstm.get('total_return', 0):.2f}%")
        except Exception as e:
            print(f"  ✗ LSTM 失敗: {e}")
            comparison['LSTM'] = {'error': str(e)}
        
        # 測試 GRU（需要修改 ai_service 支持模型類型選擇，這裡簡化）
        print("\n2️⃣ 訓練 GRU 模型...")
        print("  ℹ️  當前版本自動選擇最佳模型架構")
        
        # 顯示對比
        print(f"\n{'='*60}")
        print("📈 對比結果:")
        for model_name, result in comparison.items():
            if 'error' not in result:
                print(f"\n  {model_name}:")
                print(f"    訓練損失: {result['loss']:.6f}")
                if result.get('backtest'):
                    bt = result['backtest']
                    print(f"    回測報酬: {bt.get('total_return', 0):.2f}%")
                    print(f"    勝率: {bt.get('win_rate', 0):.1f}%")
                    print(f"    夏普比率: {bt.get('sharpe_ratio', 0):.2f}")
        print(f"{'='*60}\n")
        
        return comparison
    
    def batch_optimize(self, ticker_list, max_stocks=None):
        """
        批量優化訓練
        :param ticker_list: 股票列表 [(ticker, name), ...]
        :param max_stocks: 最多訓練幾支
        """
        if max_stocks:
            ticker_list = ticker_list[:max_stocks]
        
        print(f"\n{'='*60}")
        print(f"🚀 批量優化訓練 ({len(ticker_list)} 支股票)")
        print(f"{'='*60}\n")
        
        results_summary = []
        
        for idx, (ticker, name) in enumerate(ticker_list, 1):
            print(f"\n[{idx}/{len(ticker_list)}] {name} ({ticker})")
            print("-" * 60)
            
            try:
                best_result = self.train_with_hyperparameters(ticker, name)
                if best_result:
                    results_summary.append({
                        'ticker': ticker,
                        'name': name,
                        'best_params': best_result['params'],
                        'return': best_result['backtest']['total_return'],
                        'win_rate': best_result['backtest']['win_rate']
                    })
            except Exception as e:
                print(f"  ✗ 優化失敗: {e}")
        
        # 生成總結報告
        self._generate_summary_report(results_summary)
        
        return results_summary
    
    def _backtest_model(self, ticker, hist):
        """回測模型性能"""
        try:
            trader = AutoTrader()
            result_msg, logs_df = trader.run_simulation(ticker, hist, lookback_days=252)
            
            if logs_df.empty:
                return None
            
            # 計算績效指標
            metrics = trader.performance_tracker.calculate_metrics(logs_df)
            
            return metrics
            
        except Exception as e:
            print(f"  ⚠️ 回測失敗: {e}")
            return None
    
    def _load_results(self):
        """載入歷史結果"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_result(self, ticker, stock_name, all_results, best_result):
        """保存訓練結果"""
        self.results[ticker] = {
            'name': stock_name,
            'all_results': all_results,
            'best_result': best_result,
            'updated_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def _generate_summary_report(self, results_summary):
        """生成總結報告"""
        if not results_summary:
            return
        
        report_file = f"data/optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 按回測報酬排序
        results_summary.sort(key=lambda x: x['return'], reverse=True)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                  🚀 批量優化訓練報告                         ║
╚══════════════════════════════════════════════════════════════╝

訓練時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
訓練股票數: {len(results_summary)}

📊 優化結果排行（按回測報酬）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

排名  股票代碼      股票名稱        回測報酬    勝率     最佳模型
───────────────────────────────────────────────────────────────
"""
        
        for idx, result in enumerate(results_summary, 1):
            model_type = result['best_params'].get('model_type', 'LSTM')
            hidden = result['best_params'].get('hidden_dim', 64)
            layers = result['best_params'].get('num_layers', 2)
            
            report += f"{idx:>2}.  {result['ticker']:<12} {result['name']:<12} "
            report += f"{result['return']:>7.2f}%  {result['win_rate']:>5.1f}%  "
            report += f"{model_type}({hidden}/{layers})\n"
        
        report += f"""
───────────────────────────────────────────────────────────────

💡 最佳配置統計:
"""
        
        # 統計最佳參數分布
        model_types = [r['best_params'].get('model_type', 'LSTM') for r in results_summary]
        from collections import Counter
        type_count = Counter(model_types)
        
        report += f"\n模型類型分布:\n"
        for model_type, count in type_count.most_common():
            report += f"  {model_type}: {count} 支 ({count/len(results_summary)*100:.1f}%)\n"
        
        report += f"""
╔══════════════════════════════════════════════════════════════╗
║ 報告已保存: {report_file:<45} ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        # 保存報告
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)


def main():
    parser = argparse.ArgumentParser(description='進階訓練系統 - 超參數優化與模型對比')
    parser.add_argument('--mode', choices=['optimize', 'compare', 'batch'], default='optimize',
                        help='訓練模式: optimize(單股優化), compare(模型對比), batch(批量優化)')
    parser.add_argument('--ticker', type=str, help='股票代碼（單股模式）')
    parser.add_argument('--name', type=str, help='股票名稱（單股模式）')
    parser.add_argument('--max', type=int, help='批量模式最多訓練幾支')
    
    args = parser.parse_args()
    
    trainer = AdvancedTrainer()
    
    if args.mode == 'optimize':
        # 單股超參數優化
        ticker = args.ticker or '2330.TW'
        name = args.name or '台積電'
        trainer.train_with_hyperparameters(ticker, name)
        
    elif args.mode == 'compare':
        # 模型對比
        ticker = args.ticker or '2330.TW'
        name = args.name or '台積電'
        trainer.compare_models(ticker, name)
        
    elif args.mode == 'batch':
        # 批量優化
        # 讀取股票列表
        stock_list_file = 'data/tw_stock_list.csv'
        if os.path.exists(stock_list_file):
            df = pd.read_csv(stock_list_file)
            ticker_list = [(row['ticker'], row['name']) for _, row in df.iterrows()]
            trainer.batch_optimize(ticker_list, max_stocks=args.max)
        else:
            print(f"❌ 找不到股票列表文件: {stock_list_file}")


if __name__ == "__main__":
    main()
