import json
import os
import math
import pandas as pd
import datetime
import yfinance as yf
import pytz

class PaperAccount:
    def __init__(self, data_file='data/account_data.json', initial_balance=1000000):
        self.data_file = data_file
        self.initial_balance = initial_balance
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.initial_balance)
                    self.portfolio = data.get('portfolio', {})
                    self.history = data.get('history', [])
                    self.last_dividend_check = data.get('last_dividend_check', datetime.datetime.now().strftime("%Y-%m-%d"))
            except:
                self.reset_account()
        else:
            self.reset_account()

    def reset_account(self):
        self.balance = self.initial_balance
        self.portfolio = {}  # 格式: {"Symbol": {"qty": 100, "avg_price": 50.0}}
        self.history = []
        self.last_dividend_check = datetime.datetime.now().strftime("%Y-%m-%d")
        self.save_data()

    def save_data(self):
        data = {
            'balance': self.balance,
            'portfolio': self.portfolio,
            'history': self.history,
            'last_dividend_check': self.last_dividend_check
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def buy(self, symbol, price, qty):
        cost = price * qty
        # 台股手續費 0.1425%
        fee = max(20, int(cost * 0.001425))
        total_cost = cost + fee
        
        if total_cost > self.balance:
            return False, f"餘額不足 (需 ${total_cost}, 含手續費 ${fee})"
        
        self.balance -= total_cost
        
        # 更新持倉
        if symbol in self.portfolio:
            current_qty = self.portfolio[symbol]['qty']
            current_avg = self.portfolio[symbol]['avg_price']
            new_qty = current_qty + qty
            # 平均成本計算包含手續費
            new_avg = ((current_qty * current_avg) + total_cost) / new_qty
            self.portfolio[symbol] = {'qty': new_qty, 'avg_price': new_avg}
        else:
            # 成本價 = 總花費 / 股數
            self.portfolio[symbol] = {'qty': qty, 'avg_price': float(total_cost / qty)}
            
        # 記錄交易
        self.log_transaction("BUY", symbol, price, qty, fee=fee)
        self.save_data()
        return True, f"成功以 {price} 買入 {qty} 股 {symbol} (手續費 ${fee})"

    def sell(self, symbol, price, qty):
        if symbol not in self.portfolio or self.portfolio[symbol]['qty'] < qty:
            return False, "持倉不足"
        
        revenue = price * qty
        # 手續費 0.1425%
        fee = max(20, int(revenue * 0.001425))
        # 證交稅 0.3%
        tax = int(revenue * 0.003)
        
        net_income = revenue - fee - tax
        self.balance += net_income
        
        # 更新持倉
        self.portfolio[symbol]['qty'] -= qty
        if self.portfolio[symbol]['qty'] == 0:
            del self.portfolio[symbol]
            
        # 記錄交易 (real_amount 為實領)
        self.log_transaction("SELL", symbol, price, qty, real_amount=net_income, fee=fee, tax=tax)
        self.save_data()
        return True, f"成功以 {price} 賣出 {qty} 股 {symbol} (扣除手續費 ${fee}, 證交稅 ${tax}, 實得 ${net_income})"


    def check_dividends(self):
        """檢查並發放股息"""
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if self.last_dividend_check >= today_str:
            return [], "今日已檢查過股息"

        total_dividends = 0
        dividend_msgs = []
        
        # 轉換上次檢查日期為 datetime 物件
        last_check_date = datetime.datetime.strptime(self.last_dividend_check, "%Y-%m-%d")
        
        for symbol, data in self.portfolio.items():
            try:
                stock = yf.Ticker(symbol)
                # 獲取最近配息紀錄
                # 這裡簡單抓取最近 1 個月的歷史，然後過濾日期
                hist = stock.history(period="1mo")
                if 'Dividends' in hist.columns:
                    dividends = hist[hist['Dividends'] > 0]['Dividends']
                    
                    for date, div_amount in dividends.items():
                        # 將 pandas Timestamp 轉為本地 date 比較
                        div_date = date.to_pydatetime().date()
                        # 如果配息日在上次檢查之後 (且已發生)
                        if div_date > last_check_date.date() and div_date <= datetime.date.today():
                            qty = data['qty']
                            gross_payout = div_amount * qty
                            
                            # --- 實領金額計算 (模擬現實稅費) ---
                            deductions_str = ""
                            net_payout = gross_payout
                            
                            is_tw = symbol.endswith('.TW') or symbol.endswith('.TWO')
                            
                            if is_tw:
                                # 1. 二代健保 (單筆 >= 20000 扣 2.11%)
                                health_fee = 0
                                if gross_payout >= 20000:
                                    health_fee = math.floor(gross_payout * 0.0211)
                                
                                # 2. 匯費 (模擬 10 元，若金額太小則不扣到負)
                                transfer_fee = 10 if gross_payout > 10 else 0
                                
                                net_payout = gross_payout - health_fee - transfer_fee
                                if health_fee > 0:
                                    deductions_str += f" (含健保費 -{health_fee}"
                                if transfer_fee > 0:
                                    deductions_str += f"{', ' if health_fee>0 else ' ('}匯費 -{transfer_fee}"
                                if deductions_str:
                                    deductions_str += ")"
                                
                            else:
                                # 美股預扣稅 (30%)
                                tax = gross_payout * 0.30
                                net_payout = gross_payout - tax
                                deductions_str = f" (預扣稅 -{tax:.1f})"

                            if net_payout < 0: net_payout = 0
                            
                            # 入帳
                            self.balance += net_payout
                            total_dividends += net_payout
                            
                            # 紀錄 Log: price 欄位借放 每股股息, amount 放實領
                            self.log_transaction("DIVIDEND", symbol, div_amount, qty, real_amount=net_payout)
                            dividend_msgs.append(f"{symbol} 配息: 應發 ${gross_payout:.0f} -> 實領 ${net_payout:.0f}{deductions_str}")
            except Exception as e:
                print(f"Check dividend error for {symbol}: {e}")
                continue

        # 更新檢查日期
        self.last_dividend_check = today_str
        self.save_data()
        
        return dividend_msgs, f"股息檢查完成，共實領 ${total_dividends:.0f}"

    def log_transaction(self, action, symbol, price, qty, real_amount=None, fee=0, tax=0):
        # 對於 DIVIDEND，price 是每股股息
        if real_amount is not None:
             amount = real_amount
        else:
             amount = price * qty
        
        record = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "symbol": symbol,
            "price": price,
            "qty": qty,
            "amount": amount,
            "fee": fee,
            "tax": tax
        }
        self.history.insert(0, record) # 新的在前面

    def get_portfolio_status(self, current_prices):
        """計算當前持倉價值和損益"""
        rows = []
        total_market_value = 0
        
        for symbol, data in self.portfolio.items():
            qty = data['qty']
            avg_price = data['avg_price']
            current_price = current_prices.get(symbol, avg_price) # 如果抓不到現價就用成本價算
            market_value = qty * current_price
            profit_loss = (current_price - avg_price) * qty
            profit_loss_pct = (profit_loss / (avg_price * qty)) * 100 if avg_price > 0 else 0
            
            total_market_value += market_value
            
            rows.append({
                "代碼": symbol,
                "持有數量": qty,
                "平均成本": round(avg_price, 2),
                "現價": round(current_price, 2),
                "市值": round(market_value, 2),
                "損益": round(profit_loss, 2),
                "報酬率%": round(profit_loss_pct, 2)
            })
            
        return rows, total_market_value
