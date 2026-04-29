"""
🌟 市場級別自動訓練系統 v2.0
每日自動訓練整個市場的股票模型，讓 AI 持續進化
"""

import sys
import os

# 將專案根目錄加入 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import pandas as pd
import json
from datetime import datetime, timedelta
from services.ai_service import AIService
from utils import get_stock_data
import torch


class MarketTrainingSystem:
    """市場級別訓練系統"""

    def __init__(self):
        self.service = AIService()
        self.training_log_file = "data/training_log.json"
        self.training_history = self.load_training_history()

    def load_training_history(self):
        """載入訓練歷史"""
        if os.path.exists(self.training_log_file):
            try:
                with open(self.training_log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_training_history(self):
        """保存訓練歷史"""
        with open(self.training_log_file, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, ensure_ascii=False, indent=2)

    def should_train_today(self, ticker):
        """
        判斷今天是否需要訓練此股票

        策略：
        - 新股票：立即訓練
        - 已訓練：每週訓練一次
        - 重要股票：每日訓練
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # 檢查是否有訓練記錄
        if ticker not in self.training_history:
            return True, "新股票，首次訓練"

        last_train_date = self.training_history[ticker].get("last_train_date")
        if not last_train_date:
            return True, "無訓練日期記錄"

        # 計算距離上次訓練的天數
        last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
        days_since_train = (datetime.now() - last_date).days

        # 重要股票列表（市值前 50 大或交易熱門）
        important_tickers = self.get_important_tickers()

        if ticker in important_tickers:
            # 重要股票：每日訓練
            if days_since_train >= 1:
                return True, f"重要股票，距上次訓練 {days_since_train} 天"
        else:
            # 一般股票：每週訓練
            if days_since_train >= 7:
                return True, f"一般股票，距上次訓練 {days_since_train} 天"

        return False, f"最近才訓練過 ({last_train_date})"

    def get_important_tickers(self):
        """
        獲取重要股票清單

        包含：
        - 台灣 50 成分股
        - 交易量前 100 大
        - ETF
        """
        important = [
            # 台灣 50 權重股
            "2330.TW",
            "2317.TW",
            "2454.TW",
            "2881.TW",
            "2882.TW",
            "2412.TW",
            "2308.TW",
            "2886.TW",
            "2891.TW",
            "2303.TW",
            "2002.TW",
            "1301.TW",
            "1303.TW",
            "2884.TW",
            "2892.TW",
            "2379.TW",
            "2357.TW",
            "3711.TW",
            "2382.TW",
            "2885.TW",
            "2207.TW",
            "3008.TW",
            "2395.TW",
            "2301.TW",
            "2409.TW",
            # 熱門 ETF
            "0050.TW",
            "0056.TW",
            "006208.TW",
            "00878.TW",
            "00713.TW",
            "00631L.TW",
            "00632R.TW",
            "00757.TW",
            "00881.TW",
            "00882.TW",
        ]
        return set(important)

    def train_single_stock(self, ticker, stock_name, force=False):
        """
        訓練單一股票

        參數:
            ticker: 股票代碼
            stock_name: 股票名稱
            force: 強制訓練（忽略時間檢查）

        返回:
            (成功, 訊息, 耗時)
        """
        # 檢查是否需要訓練
        if not force:
            should_train, reason = self.should_train_today(ticker)
            if not should_train:
                return False, f"跳過: {reason}", 0

        try:
            # 獲取股票數據
            _, hist, _, _ = get_stock_data(ticker)

            if hist.empty or len(hist) < 200:
                msg = f"數據不足 (僅{len(hist)}天)"
                self.training_history[ticker] = {
                    "last_train_date": datetime.now().strftime("%Y-%m-%d"),
                    "status": "failed",
                    "reason": msg,
                }
                return False, msg, 0

            # 訓練模型
            start_time = time.time()

            # 使用遷移學習加速訓練
            result, final_loss = self.service.train_deep_learning_model(
                ticker, hist, use_transfer_learning=True  # 啟用遷移學習
            )

            elapsed = time.time() - start_time

            # 記錄訓練歷史
            self.training_history[ticker] = {
                "last_train_date": datetime.now().strftime("%Y-%m-%d"),
                "last_train_time": datetime.now().strftime("%H:%M:%S"),
                "status": "success",
                "final_loss": float(final_loss),
                "training_time": elapsed,
                "data_days": len(hist),
                "stock_name": stock_name,
            }

            return True, f"Loss: {final_loss:.6f}", elapsed

        except Exception as e:
            msg = str(e)[:100]
            self.training_history[ticker] = {
                "last_train_date": datetime.now().strftime("%Y-%m-%d"),
                "status": "error",
                "reason": msg,
            }
            return False, msg, 0

    def train_market(self, max_stocks=None, priority="important", save_interval=1):
        """
        訓練整個市場

        參數:
            max_stocks: 最多訓練幾支（None=全部）
            priority: 優先順序 ('important'=重要股票優先, 'all'=全部平等, 'untrained'=未訓練優先)
            save_interval: 每訓練幾支就保存一次歷史記錄 (預設 1，即每支都存)
        """
        print("=" * 80)
        print("🚀 市場級別自動訓練系統 v2.0")
        print("=" * 80)
        print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 檢測硬體
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU 加速: {gpu_name}")
        else:
            print("💻 CPU 訓練模式")

        # 讀取股票列表
        try:
            df = pd.read_csv("data/tw_stock_list.csv")
        except:
            df = pd.read_csv("tw_stock_list.csv")

        # 過濾：只訓練有數據的股票
        tickers = df["Ticker"].tolist()

        # 根據優先順序排序
        if priority == "important":
            # 重要股票優先
            important = self.get_important_tickers()
            important_list = [t for t in tickers if t in important]
            other_list = [t for t in tickers if t not in important]
            tickers = important_list + other_list
            print(f"📊 優先訓練 {len(important_list)} 支重要股票")

        elif priority == "untrained":
            # 未訓練優先
            untrained = [t for t in tickers if t not in self.training_history]
            trained = [t for t in tickers if t in self.training_history]
            tickers = untrained + trained
            print(f"🆕 優先訓練 {len(untrained)} 支未訓練股票")

        if max_stocks:
            tickers = tickers[:max_stocks]

        print(f"📈 目標數量: {len(tickers)} 支股票")
        print(f"💾 自動保存: 每 {save_interval} 支")
        print()

        # 統計變數
        success_count = 0
        skip_count = 0
        fail_count = 0
        total_time = 0

        for idx, ticker in enumerate(tickers, 1):
            stock_name = (
                df[df["Ticker"] == ticker]["名稱"].values[0]
                if len(df[df["Ticker"] == ticker]) > 0
                else ticker
            )

            print(f"\n{'='*80}")
            print(f"[{idx}/{len(tickers)}] {ticker} ({stock_name})")
            print(f"{'='*80}")

            # 訓練
            success, msg, elapsed = self.train_single_stock(ticker, stock_name)

            if success:
                print(f"✅ 訓練成功！{msg} | 耗時: {elapsed:.1f}秒")
                success_count += 1
                total_time += elapsed
            elif "跳過" in msg:
                print(f"⏭️  {msg}")
                skip_count += 1
            else:
                print(f"❌ 訓練失敗: {msg}")
                fail_count += 1

            # 統計資訊
            trained_count = success_count + fail_count
            avg_time = total_time / success_count if success_count > 0 else 0
            remaining = (len(tickers) - idx) * avg_time if avg_time > 0 else 0

            print(f"\n📊 進度統計:")
            print(
                f"   成功: {success_count} | 跳過: {skip_count} | 失敗: {fail_count} | 剩餘: {len(tickers) - idx}"
            )
            if success_count > 0:
                print(
                    f"   平均耗時: {avg_time:.1f}秒/支 | 預估剩餘: {remaining/60:.1f}分鐘"
                )

            # 定期保存
            if idx % save_interval == 0:
                self.save_training_history()
                print(f"   💾 已保存訓練歷史 (第 {idx} 支)")

            # GPU 降溫（每訓練 5 支休息一次）
            if success_count > 0 and success_count % 5 == 0 and idx < len(tickers):
                print(f"\n🧊 GPU 降溫中...")
                time.sleep(5)

        # 最終保存
        self.save_training_history()

        # 最終統計
        print("\n" + "=" * 80)
        print("🎉 市場訓練完成！")
        print("=" * 80)
        print(f"總數量: {len(tickers)} 支")
        print(f"成功: {success_count} 支 ({success_count/len(tickers)*100:.1f}%)")
        print(f"跳過: {skip_count} 支 ({skip_count/len(tickers)*100:.1f}%)")
        print(f"失敗: {fail_count} 支 ({fail_count/len(tickers)*100:.1f}%)")
        if success_count > 0:
            print(f"總耗時: {total_time/60:.1f} 分鐘")
            print(f"平均速度: {total_time/success_count:.1f} 秒/支")
        print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 生成訓練報告
        self.generate_report()

    def generate_report(self):
        """生成優化版市場訓練診斷報告"""
        if not self.training_history:
            return

        report_file = (
            f"data/training_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )

        # 準備表格數據
        success_items = [
            (k, v)
            for k, v in self.training_history.items()
            if v.get("status") == "success"
        ]
        # 按損失值排序（預測越準的排前面）
        success_items.sort(key=lambda x: x[1].get("final_loss", 999))

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("🤖 AI 股票模型訓練 & 市場診斷報告\n")
            f.write("=" * 100 + "\n")
            f.write(f"📅 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 總體統計
            total = len(self.training_history)
            success = len(success_items)
            f.write(
                f"📊 統計概要: 總掃描 {total} 支 | 成功訓練 {success} 支 | 失敗 {total-success} 支\n\n"
            )

            # 定義表頭
            header = f"{'股票代碼':<12} {'名稱':<10} {'預測可靠度':<12} {'當前趨勢':<12} {'波動風險':<10} {'Loss (誤差)':<12} {'建議建議'}\n"
            f.write(header)
            f.write("-" * 100 + "\n")

            for ticker, info in success_items:
                name = info.get("stock_name", "")
                loss = info.get("final_loss", 0)

                # 1. 轉換可靠度 (基於 Loss)
                if loss < 0.0005:
                    reliability = "💎 極高"
                elif loss < 0.002:
                    reliability = "✅ 高"
                elif loss < 0.005:
                    reliability = "⚠️ 中"
                else:
                    reliability = "❓ 低"

                # 2. 獲取趨勢資訊 (嘗試從快取或重新判定)
                # 這裡做一個簡單的邏輯判斷，實際可從技術指標模組獲取
                trend = "🔄 盤整"
                risk = "🟢 低"
                advice = "觀察"

                # 根據 Loss 賦予簡單建議
                if reliability in ["💎 極高", "✅ 高"]:
                    advice = "策略可參考"
                else:
                    advice = "僅供輔助"

                # 格式化輸出
                f.write(
                    f"{ticker:<12} {name:<10} {reliability:<12} {trend:<12} {risk:<10} {loss:<14.6f} {advice}\n"
                )

            f.write("-" * 100 + "\n")
            f.write(
                "💡 說明：Loss 越低代表 AI 對該標的歷史走勢掌握度越高；可靠度「極高」者，其 AI 預測軌跡較具參考價值。\n"
            )
            f.write("=" * 100 + "\n")

        print(f"\n📄 診斷報告已生成: {report_file}")


def main():
    """主程式入口"""
    import argparse

    parser = argparse.ArgumentParser(description="市場級別自動訓練系統")
    parser.add_argument(
        "--max", type=int, default=None, help="最多訓練幾支股票（預設：全部）"
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="important",
        choices=["important", "all", "untrained"],
        help="優先順序（important=重要股票優先, all=全部平等, untrained=未訓練優先）",
    )
    parser.add_argument("--force", action="store_true", help="強制訓練（忽略時間檢查）")

    args = parser.parse_args()

    # 建立訓練系統
    trainer = MarketTrainingSystem()

    # 開始訓練
    trainer.train_market(max_stocks=args.max, priority=args.priority)


if __name__ == "__main__":
    main()
