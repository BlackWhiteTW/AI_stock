"""
📊 績效追蹤系統 v1.0
記錄所有交易並計算完整的績效指標
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class PerformanceTracker:
    """績效追蹤器"""
    
    def __init__(self, log_file='data/trade_log.json'):
        self.log_file = log_file
        self.trades = self._load_trades()
        
    def log_trade(self, ticker, action, price, quantity, date, reason=''):
        """
        記錄一筆交易
        :param ticker: 股票代碼
        :param action: 'BUY' or 'SELL'
        :param price: 成交價格
        :param quantity: 成交數量
        :param date: 交易日期
        :param reason: 交易原因
        """
        trade = {
            'date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date),
            'ticker': ticker,
            'action': action,
            'price': float(price),
            'quantity': int(quantity),
            'amount': float(price * quantity),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        self._save_trades()
        
    def calculate_metrics(self, account_history=None):
        """
        計算完整績效指標
        :param account_history: 帳戶歷史資料 DataFrame
        :return: dict of metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        df = pd.DataFrame(self.trades)
        
        # 配對買賣交易
        positions = self._match_trades(df)
        
        if not positions:
            return self._empty_metrics()
        
        pos_df = pd.DataFrame(positions)
        
        # 基本指標
        total_trades = len(pos_df)
        winning_trades = len(pos_df[pos_df['profit'] > 0])
        losing_trades = len(pos_df[pos_df['profit'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = pos_df['profit'].sum()
        avg_profit = pos_df[pos_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = pos_df[pos_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        # 獲利因子 (Profit Factor)
        gross_profit = pos_df[pos_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(pos_df[pos_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 平均持有天數
        avg_hold_days = pos_df['hold_days'].mean() if 'hold_days' in pos_df.columns else 0
        
        # 最大連勝/連敗
        max_consecutive_wins = self._max_consecutive(pos_df['profit'] > 0)
        max_consecutive_losses = self._max_consecutive(pos_df['profit'] < 0)
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate * 100, 2),
            'total_profit': round(total_profit, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_hold_days': round(avg_hold_days, 1),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
        
        # 如果有帳戶歷史，計算進階指標
        if account_history is not None and not account_history.empty:
            advanced = self._calculate_advanced_metrics(account_history)
            metrics.update(advanced)
        
        return metrics
    
    def _calculate_advanced_metrics(self, account_history):
        """計算進階指標（夏普比率、最大回撤等）"""
        if account_history.empty or 'total_assets' not in account_history.columns:
            return {}
        
        # 確保有日期索引
        if 'date' in account_history.columns:
            account_history = account_history.set_index('date')
        
        # 計算日報酬率
        returns = account_history['total_assets'].pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # 總報酬率
        initial_value = account_history['total_assets'].iloc[0]
        final_value = account_history['total_assets'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # 年化報酬率（假設252個交易日）
        days = len(account_history)
        annual_return = ((final_value / initial_value) ** (252 / days) - 1) * 100 if days > 0 else 0
        
        # 夏普比率（假設無風險利率2%）
        risk_free_rate = 0.02 / 252  # 日化
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 波動率（年化）
        volatility = returns.std() * np.sqrt(252) * 100
        
        return {
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'volatility': round(volatility, 2)
        }
    
    def _match_trades(self, df):
        """配對買賣交易，計算每次完整交易的損益"""
        positions = []
        
        for ticker in df['ticker'].unique():
            ticker_trades = df[df['ticker'] == ticker].sort_values('date')
            
            buy_queue = []
            
            for _, trade in ticker_trades.iterrows():
                if trade['action'] == 'BUY':
                    buy_queue.append(trade)
                elif trade['action'] == 'SELL' and buy_queue:
                    # FIFO 配對
                    buy_trade = buy_queue.pop(0)
                    
                    # 計算損益
                    buy_amount = buy_trade['price'] * buy_trade['quantity']
                    sell_amount = trade['price'] * trade['quantity']
                    profit = sell_amount - buy_amount
                    profit_pct = (profit / buy_amount) * 100
                    
                    # 持有天數
                    buy_date = pd.to_datetime(buy_trade['date'])
                    sell_date = pd.to_datetime(trade['date'])
                    hold_days = (sell_date - buy_date).days
                    
                    positions.append({
                        'ticker': ticker,
                        'buy_date': buy_trade['date'],
                        'sell_date': trade['date'],
                        'buy_price': buy_trade['price'],
                        'sell_price': trade['price'],
                        'quantity': trade['quantity'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'hold_days': hold_days
                    })
        
        return positions
    
    def _max_consecutive(self, condition):
        """計算最大連續次數"""
        if len(condition) == 0:
            return 0
        
        max_count = 0
        current_count = 0
        
        for val in condition:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def _empty_metrics(self):
        """空指標"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_hold_days': 0
        }
    
    def _load_trades(self):
        """從文件載入交易記錄"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_trades(self):
        """儲存交易記錄到文件"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, ensure_ascii=False, indent=2)
    
    def generate_report(self, account_history=None, output_file=None):
        """
        生成績效報告
        :param account_history: 帳戶歷史
        :param output_file: 輸出文件路徑
        :return: str (報告內容)
        """
        metrics = self.calculate_metrics(account_history)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    📊 交易績效報告                           ║
╚══════════════════════════════════════════════════════════════╝

📈 基本指標
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
總交易次數:           {metrics['total_trades']:>10}
獲利交易:             {metrics['winning_trades']:>10}
虧損交易:             {metrics['losing_trades']:>10}
勝率:                 {metrics['win_rate']:>9.2f}%

💰 損益分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
總損益:               {metrics['total_profit']:>10,.2f}
平均獲利:             {metrics['avg_profit']:>10,.2f}
平均虧損:             {metrics['avg_loss']:>10,.2f}
獲利因子:             {metrics['profit_factor']:>10.2f}

📊 交易分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
平均持有天數:         {metrics['avg_hold_days']:>10.1f}
最大連勝:             {metrics['max_consecutive_wins']:>10}
最大連敗:             {metrics['max_consecutive_losses']:>10}
"""
        
        # 進階指標
        if 'total_return' in metrics:
            report += f"""
🎯 進階指標
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
總報酬率:             {metrics['total_return']:>9.2f}%
年化報酬率:           {metrics['annual_return']:>9.2f}%
夏普比率:             {metrics['sharpe_ratio']:>10.2f}
最大回撤:             {metrics['max_drawdown']:>9.2f}%
波動率:               {metrics['volatility']:>9.2f}%
"""
        
        report += f"""
╔══════════════════════════════════════════════════════════════╗
║ 報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):>45} ║
╚══════════════════════════════════════════════════════════════╝
"""
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def get_trade_summary_by_ticker(self):
        """按股票統計交易績效"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        positions = self._match_trades(df)
        
        if not positions:
            return pd.DataFrame()
        
        pos_df = pd.DataFrame(positions)
        
        summary = pos_df.groupby('ticker').agg({
            'profit': ['count', 'sum', 'mean'],
            'profit_pct': 'mean',
            'hold_days': 'mean'
        }).round(2)
        
        summary.columns = ['交易次數', '總損益', '平均損益', '平均獲利率%', '平均持有天數']
        
        return summary.sort_values('總損益', ascending=False)


# 測試
if __name__ == "__main__":
    tracker = PerformanceTracker()
    
    # 模擬交易
    tracker.log_trade('2330.TW', 'BUY', 580, 1000, datetime.now(), 'AI買入信號')
    tracker.log_trade('2330.TW', 'SELL', 600, 1000, datetime.now(), 'AI賣出信號')
    
    # 計算指標
    metrics = tracker.calculate_metrics()
    print(metrics)
    
    # 生成報告
    report = tracker.generate_report()
    print(report)
