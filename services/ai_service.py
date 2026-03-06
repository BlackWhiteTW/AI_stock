import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import time
import os
import gc
import json
import datetime
import traceback
from utils import get_stock_data
from .ai_model_defs import LSTMModel
from . import tech_analysis
from .stock_strategy import StockStrategy
from .market_sentiment import MarketSentimentAnalyzer

try:
    from .news_analyzer import NewsAnalyzer

    NEWS_AVAILABLE = True
except:
    NEWS_AVAILABLE = False

try:
    from .weekly_news_collector import WeeklyNewsCollector

    NEWS_COLLECTOR_AVAILABLE = True
except:
    NEWS_COLLECTOR_AVAILABLE = False


class AIService:
    """
    🚀 高性能 AI 預測服務 v5.4 (2026 旗艦版)
    專注於 Residual Attention LSTM 與 Alpha 特徵融合
    🆕 整合周期性新聞情緒特徵
    """

    MODEL_VERSION = "5.4-LSTM-Attention-News"

    def __init__(self):
        self.models = {}  # Metadata
        self.loaded_models = {}  # LRU Cache
        self.MAX_CACHE_SIZE = 15

        self.sentiment_analyzer = MarketSentimentAnalyzer()

        # 🆕 v5.4 新增：周期性新聞收集器
        if NEWS_COLLECTOR_AVAILABLE:
            try:
                self.news_collector = WeeklyNewsCollector()
                print("✅ 周期性新聞收集器已啟用")
            except Exception as e:
                print(f"⚠️ 新聞收集器初始化失敗: {e}")
                self.news_collector = None
        else:
            self.news_collector = None

        self.lookback = 60
        self.model_dir = "data/saved_models"
        self.prediction_file = "data/daily_predictions.json"
        self.base_model_path = "data/saved_models/base_model.pth"

        # 🆕 v5.4 完整 30 個核心與 Alpha 特徵（含新聞情緒）
        self.feature_columns = [
            # 基礎價量 (5)
            "Close",
            "Open",
            "High",
            "Low",
            "Volume",
            # 技術指標 (6)
            "RSI",
            "MACD",
            "K",
            "D",
            "ATR",
            "Volatility",
            # 市場情緒 (5)
            "MFI",
            "ADX",
            "Price_Position",
            "Volume_Anomaly",
            "Market_Sentiment_Score",
            # Alpha 動能 (4)
            "Price_Velocity",
            "Price_Acceleration",
            "OBV",
            "BIAS_20",
            # 財務與相對強度 (6)
            "Relative_Strength",
            "Dist_MA20",
            "fin_roe_normalized",
            "fin_debt_ratio_normalized",
            "fin_pe_ratio_normalized",
            "fin_dividend_yield_normalized",
            # 🆕 新聞情緒特徵 (4)
            "news_sentiment_7d",
            "news_count_7d",
            "news_intensity",
            "news_sentiment_change",
        ]

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.strategy = StockStrategy()
        self._load_all_models()

    def _fetch_representative_stocks(self, count=200):
        """🌐 從網路與本地清單動態獲取各產業代表性標的 (目標 200 支)"""
        print(f"🌐 正在構建多元化訓練標的名單 (目標: {count} 支)...")
        stock_list = []

        try:
            from .yahoo_tw_scraper import get_yahoo_scraper

            scraper = get_yahoo_scraper()

            # 1. 核心權值股 (0050 + 0051 + 0056)
            for etf in ["0050.TW", "0051.TW", "0056.TW"]:
                holding_data = scraper.get_holding(etf)
                if holding_data and "top_holdings" in holding_data:
                    for item in holding_data["top_holdings"]:
                        raw_id = (
                            item.get("代號")
                            or item.get("股票代號")
                            or item.get("股票", "")
                        )
                        clean_id = "".join(filter(str.isdigit, str(raw_id)))
                        if len(clean_id) == 4:  # 只選一般個股，排除權證
                            ticker = f"{clean_id}.TW"
                            if ticker not in stock_list:
                                stock_list.append(ticker)

            # 2. 產業均衡補充 (從本地 CSV 各產業分散選取)
            if len(stock_list) < count and os.path.exists("data/tw_stock_list.csv"):
                try:
                    df_all = pd.read_csv("data/tw_stock_list.csv")
                    # 清洗數據：只留 4 位數代號
                    df_all["Ticker_Len"] = df_all["代碼"].astype(str).str.len()
                    df_clean = df_all[df_all["Ticker_Len"] == 4].copy()

                    # 按產業分組，每組取前 N 支 (通常 CSV 已按規模排序)
                    industries = df_clean["產業"].unique()
                    per_industry = max(
                        1, (count - len(stock_list)) // len(industries) + 1
                    )

                    for ind in industries:
                        ind_stocks = df_clean[df_clean["產業"] == ind].head(
                            per_industry
                        )
                        for _, row in ind_stocks.iterrows():
                            ticker = str(row["Ticker"])
                            if ticker not in stock_list:
                                stock_list.append(ticker)
                            if len(stock_list) >= count:
                                break
                        if len(stock_list) >= count:
                            break
                except Exception as e:
                    print(f"⚠️ 產業抽樣失敗: {e}")

        except Exception as e:
            print(f"⚠️ 獲取代表性標的時發生錯誤: {e}")

        # 3. 確保至少有保底清單
        base_stocks = [
            "2330.TW",
            "2317.TW",
            "2454.TW",
            "2303.TW",
            "2308.TW",
            "2603.TW",
            "2881.TW",
            "2882.TW",
        ]
        for s in base_stocks:
            if s not in stock_list:
                stock_list.insert(0, s)

        final_list = stock_list[:count]
        print(f"✅ 已選定 {len(final_list)} 支各產業代表性標的")
        return final_list

    def _load_all_models(self):
        """掃描本地磁碟中的模型檔案"""
        self.models = {}
        if not os.path.exists(self.model_dir):
            return
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pth"):
                ticker = filename.replace(".pth", "")
                self.models[ticker] = {"last_loss": 0.0}

    def _get_model_instance(self, ticker):
        """[LRU Cache] 從記憶體或硬碟加載模型"""
        if ticker in self.loaded_models:
            val = self.loaded_models.pop(ticker)
            self.loaded_models[ticker] = val
            return val["model"], val["scaler"]

        path = os.path.join(self.model_dir, f"{ticker}.pth")
        if not os.path.exists(path):
            return None, None

        try:
            # 兼容性加載
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            input_dim = checkpoint.get("input_dim", len(self.feature_columns))
            hidden_dim = checkpoint.get("hidden_dim", 256)
            num_layers = checkpoint.get("num_layers", 3)

            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval()

            if len(self.loaded_models) >= self.MAX_CACHE_SIZE:
                del self.loaded_models[next(iter(self.loaded_models))]

            self.loaded_models[ticker] = {
                "model": model,
                "scaler": checkpoint["scaler"],
            }
            return model, checkpoint["scaler"]
        except Exception as e:
            print(f"⚠️ 模型 {ticker} 加載失敗: {e}")
            return None, None

    def prepare_lstm_data(self, df, lookback, ticker=None):
        """準備與標準化 LSTM 數據"""
        df = self._add_market_sentiment_features(df)

        # 確保特徵完整計算
        if "Price_Velocity" not in df.columns:
            df = tech_analysis.calculate_technical_indicators(
                df, ticker=ticker, include_financial=True
            )

        # 處理缺失特徵
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        data_clean = df[self.feature_columns].fillna(0).iloc[60:].copy()
        if data_clean.empty:
            data_clean = df[self.feature_columns].fillna(0).copy()

        # 高波動股票對數處理
        price_cv = data_clean["Close"].std() / data_clean["Close"].mean()
        if price_cv > 0.6:
            for feat in ["Close", "Open", "High", "Low"]:
                data_clean[feat] = np.log1p(data_clean[feat])

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data_clean.values)

        X, y = [], []
        for i in range(len(scaled_data) - lookback - 1):
            X.append(scaled_data[i : i + lookback])
            y.append(scaled_data[i + lookback, 0])  # 預測目標：Close

        return np.array(X), np.array(y), scaler

    def train_deep_learning_model(
        self, ticker, df, progress_callback=None, use_transfer_learning=True
    ):
        """
        🔥 專業級 LSTM 訓練引擎
        包含 GPU 加速、SWA、方向性損失優化
        """
        if len(df) > 3000:
            df = df.iloc[-3000:].copy()
        if len(df) < 200:
            return "⚠️ 數據不足 (至少需 200 天)", 0

        # 特徵工程與回測優化
        print(f"  📊 正在為 {ticker} 進行特徵工程...")
        df = tech_analysis.calculate_technical_indicators(
            df, ticker=ticker, include_financial=True
        )
        self.strategy.optimize_strategy_basic(ticker, df)

        X, y, scaler = self.prepare_lstm_data(df, self.lookback, ticker=ticker)

        # 數據分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 硬體偵測與優化配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 128 if torch.cuda.is_available() else 32
        num_workers = 2 if torch.cuda.is_available() else 0

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
            ),
            batch_size=batch_size * 2,
            shuffle=False,
        )

        # 模型初始化
        input_dim = X.shape[2]
        hidden_dim = 256 if len(X) > 1000 else 128
        num_layers = 3 if len(X) > 1000 else 2
        model = LSTMModel(input_dim, hidden_dim, num_layers, 1).to(device)

        # 🚀 遷移學習核心
        if use_transfer_learning and os.path.exists(self.base_model_path):
            try:
                base = torch.load(
                    self.base_model_path, map_location=device, weights_only=False
                )
                if base["input_dim"] == input_dim:
                    model.load_state_dict(base["model_state_dict"], strict=False)
                    print(f"  ✓ 已從主模型繼承權重")
            except:
                pass

        # 損失函數與優化器 (Huber + Directional Penalty)
        criterion_base = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        # SWA 設置
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.0001)
        swa_start = 70

        best_val_loss = float("inf")
        start_time = time.time()
        epochs = 100

        print(f"  🚀 開始訓練 {ticker} (Device: {device})")

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # 數據增強
                noise = torch.randn_like(batch_X) * 0.0001
                outputs = model(batch_X + noise)
                targets = batch_y.unsqueeze(1)

                # 基礎損失
                loss_mse = criterion_base(outputs, targets)

                # 🆕 方向性懲罰 (Directional Penalty)
                # 若預測漲跌方向與實際相反，額外加重懲罰
                last_closes = batch_X[:, -1, 0].unsqueeze(1)
                pred_diff = outputs - last_closes
                real_diff = targets - last_closes
                direction_mismatch = (pred_diff * real_diff) < 0
                directional_loss = (
                    torch.mean(direction_mismatch.float() * (outputs - targets) ** 2)
                    * 0.5
                )

                loss = loss_mse + directional_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # 更新 SWA
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # 驗證階段
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        out = model(bx)
                        val_loss += criterion_base(out, by.unsqueeze(1)).item()
                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_model(ticker, model, scaler, avg_val_loss, input_dim)

                if progress_callback:
                    progress_callback(
                        epoch + 1, epochs, avg_val_loss, time.time() - start_time
                    )

        # 結束訓練並切換回 CPU
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        final_model = swa_model.module.to("cpu")
        self.loaded_models[ticker] = {"model": final_model, "scaler": scaler}

        return f"✅ {ticker} 訓練完成！Best Loss: {best_val_loss:.6f}", best_val_loss

    def train_base_model(self, stock_list=None, progress_callback=None, enhanced=True):
        """
        🚀 2026 旗艦級：訓練 Transformer-LSTM 混合主模型
        利用 Parquet 快照進行高效全市場特徵學習
        """
        import torch.nn as nn
        from services.ai_model_defs import HybridTransLSTM
        from services.market_data import MarketDataManager
        
        data_manager = MarketDataManager()
        snapshot_path = 'data/market_snapshot.parquet'
        
        # 1. 高速數據載入與預處理
        if progress_callback: progress_callback(0, 100, 0, 0, ticker="載入市場快照")
        
        if os.path.exists(snapshot_path):
            full_df = data_manager.load_market_snapshot(snapshot_path)
            print(f"📦 已從 Parquet 快照載入 {len(full_df):,} 筆交易紀錄")
        else:
            return "❌ 找不到市場快照 (parquet)，請先在數據整合功能中生成。"

        # 2. 篩選代表性標的與樣本
        tickers = full_df['Ticker'].unique()
        selected_tickers = np.random.choice(tickers, min(100, len(tickers)), replace=False)
        
        all_X, all_y = [], []
        for i, ticker in enumerate(selected_tickers):
            df_ticker = full_df[full_df['Ticker'] == ticker].copy()
            if len(df_ticker) < 500: continue
            
            # 計算必要指標
            df_ticker = tech_analysis.calculate_technical_indicators(df_ticker, ticker=ticker)
            X, y, _ = self.prepare_lstm_data(df_ticker, self.lookback, ticker=ticker)
            if X is not None:
                # 抽樣以維持 VRAM 穩定性
                indices = np.random.choice(len(X), min(200, len(X)), replace=False)
                all_X.append(X[indices]); all_y.append(y[indices])
            
            if progress_callback and i % 10 == 0:
                progress_callback(i, len(selected_tickers), -1, 0, ticker=ticker)

        X_train = torch.from_numpy(np.vstack(all_X)).float().to(self.device)
        y_train = torch.from_numpy(np.vstack(all_y)).float().to(self.device).unsqueeze(-1)
        
        # 3. 初始化旗艦級混合模型
        model = HybridTransLSTM(
            input_dim=len(self.feature_columns),
            hidden_dim=128,
            num_layers=3,
            seq_len=self.lookback
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        
        # 4. 旗艦級訓練流程
        epochs = 100 if enhanced else 50
        batch_size = 256
        
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train))
            epoch_loss = 0
            for k in range(0, len(X_train), batch_size):
                idx = perm[k:k+batch_size]
                bx, by = X_train[idx], y_train[idx]
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(X_train) / batch_size)
            if progress_callback: progress_callback(epoch + 1, epochs, avg_loss, 0)
                
        # 5. 保存旗艦級模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': self.feature_columns,
            'input_dim': len(self.feature_columns),
            'model_type': 'HybridTransLSTM',
            'version': '2026.旗艦版'
        }, self.base_model_path)
        
        return f"✅ 旗艦級主模型訓練成功！(Loss: {avg_loss:.6f})"

    def predict_next_price(self, ticker, df):
        """執行價格預測"""
        model, scaler = self._get_model_instance(ticker)
        if not model or df.empty:
            return None

        try:
            df = tech_analysis.calculate_technical_indicators(df, ticker=ticker)
            features = self.feature_columns
            for col in features:
                if col not in df.columns:
                    df[col] = 0

            data = df[features].fillna(0).values[-self.lookback :]
            if len(data) < self.lookback:
                return None

            # 正規化
            data_scaled = scaler.transform(data)
            X = torch.from_numpy(np.array([data_scaled])).float()

            model.eval()
            with torch.no_grad():
                pred_scaled = model(X).item()

            # 反正規化
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = pred_scaled
            pred_real = scaler.inverse_transform(dummy)[0, 0]

            # 漲跌停限制輔助
            current_close = df["Close"].iloc[-1]
            pred_real = max(current_close * 0.9, min(current_close * 1.1, pred_real))

            # 🔮 動態誤差校準
            bias = self._calculate_recent_bias(ticker)
            pred_real += bias

            return self._adjust_price_tick(ticker, pred_real)
        except Exception as e:
            print(f"❌ 預測失敗 {ticker}: {e}")
            return None

    def predict_next_day(self, ticker, df):
        """為 AutoTrader v2 提供詳細預測資訊"""
        pred_price = self.predict_next_price(ticker, df)
        if pred_price is None:
            return None

        current_price = df["Close"].iloc[-1]
        change_pct = (pred_price - current_price) / current_price

        # 信心度估算：基於模型訓練損失
        confidence = 0.6
        if ticker in self.models:
            loss = self.models[ticker].get("last_loss", 0.1)
            if loss < 0.001:
                confidence = 0.85
            elif loss < 0.01:
                confidence = 0.75

        return {
            "predicted_price": float(pred_price),
            "predicted_change_pct": float(change_pct),
            "confidence": float(confidence),
            "current_price": float(current_price),
        }

    def _add_market_sentiment_features(self, df):
        """整合市場情緒特徵（含新聞情緒）"""
        try:
            df["MFI"] = self.sentiment_analyzer.calculate_money_flow_index(df)
            df["ADX"] = self.sentiment_analyzer.calculate_adx(df)
            df["Price_Position"] = self.sentiment_analyzer.get_price_position(df)
            sent = self.sentiment_analyzer.get_market_sentiment()
            df["Market_Sentiment_Score"] = sent["score"]
        except:
            pass

        # 🆕 整合周期性新聞情緒特徵
        if self.news_collector:
            try:
                news_features = self.news_collector.get_sentiment_features_for_training(
                    lookback_days=7
                )

                # 將新聞特徵擴展到整個 DataFrame（所有日期使用相同值）
                df["news_sentiment_7d"] = news_features.get("news_sentiment_7d", 0.0)
                df["news_count_7d"] = news_features.get("news_count_7d", 0)
                df["news_intensity"] = news_features.get("news_intensity", 0.0)

                # 計算新聞情緒變化率（較新數據 vs 較舊數據）
                if len(df) > 7:
                    # 用最近 3 天的新聞情緒來標記最後幾筆數據
                    recent_summary = self.news_collector.get_recent_news_summary(days=3)
                    old_summary = self.news_collector.get_recent_news_summary(days=7)

                    recent_sent = recent_summary.get("average_sentiment", 0.0)
                    old_sent = old_summary.get("average_sentiment", 0.0)

                    # 情緒變化率
                    sentiment_change = recent_sent - old_sent
                    df["news_sentiment_change"] = sentiment_change
                else:
                    df["news_sentiment_change"] = 0.0

                print(
                    f"✅ 已整合新聞情緒特徵: sentiment={news_features.get('news_sentiment_7d', 0):.2f}, count={news_features.get('news_count_7d', 0)}"
                )
            except Exception as e:
                print(f"⚠️ 新聞特徵整合失敗: {e}")
                # 使用預設值
                df["news_sentiment_7d"] = 0.0
                df["news_count_7d"] = 0
                df["news_intensity"] = 0.0
                df["news_sentiment_change"] = 0.0
        else:
            # 如果沒有新聞收集器，使用預設值
            df["news_sentiment_7d"] = 0.0
            df["news_count_7d"] = 0
            df["news_intensity"] = 0.0
            df["news_sentiment_change"] = 0.0

        return df

    def _calculate_recent_bias(self, ticker, lookback_days=5):
        """計算近期預測偏差以進行自動校準"""
        try:
            if not os.path.exists(self.prediction_file):
                return 0.0
            with open(self.prediction_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            target = [
                h
                for h in history
                if h.get("ticker") == ticker and h.get("actual_price")
            ]
            if not target:
                return 0.0
            target.sort(key=lambda x: x["date"], reverse=True)
            recent = target[:lookback_days]
            if not recent:
                return 0.0
            return sum(
                (r["actual_price"] - r["predicted_price"]) for r in recent
            ) / len(recent)
        except:
            return 0.0

    def _adjust_price_tick(self, ticker, price):
        """根據台股規則校正價格檔位"""
        try:
            if str(ticker).startswith("00"):
                return round(price, 2) if price < 50 else round(price * 20) / 20
            if price < 10:
                return round(price, 2)
            elif price < 50:
                return round(price * 20) / 20
            elif price < 100:
                return round(price, 1)
            elif price < 500:
                return round(price * 2) / 2
            elif price < 1000:
                return round(price)
            return round(price / 5) * 5
        except:
            return price

    def run_morning_predictions(self):
        """開盤前批量預測紀錄"""
        ticker_list = list(self.models.keys())
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        results = []
        for ticker in ticker_list:
            try:
                _, hist, _, _ = get_stock_data(ticker)
                pred = self.predict_next_price(ticker, hist)
                if pred:
                    base = float(hist["Close"].iloc[-1])
                    results.append(
                        {
                            "date": today_str,
                            "ticker": ticker,
                            "base_price": base,
                            "predicted_price": float(pred),
                            "expected_change_pct": round((pred - base) / base * 100, 2),
                            "actual_price": None,
                            "accuracy": "等待收盤",
                        }
                    )
            except:
                continue

        # 讀取並合併舊紀錄
        all_recs = []
        if os.path.exists(self.prediction_file):
            try:
                with open(self.prediction_file, "r", encoding="utf-8") as f:
                    all_recs = json.load(f)
            except:
                pass
        all_recs = [r for r in all_recs if r["date"] != today_str] + results
        with open(self.prediction_file, "w", encoding="utf-8") as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=4)
        return len(results)

    def validate_afternoon_results(self):
        """收盤後驗證準確度"""
        if not os.path.exists(self.prediction_file):
            return 0, []
        with open(self.prediction_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        count = 0
        for r in records:
            if r["date"] == today_str and not r.get("actual_price"):
                try:
                    _, hist, _, _ = get_stock_data(r["ticker"])
                    actual = float(hist["Close"].iloc[-1])
                    r["actual_price"] = actual
                    diff = abs(r["predicted_price"] - actual) / actual * 100
                    r["accuracy"] = (
                        "⭐⭐⭐" if diff < 1.0 else ("⭐" if diff < 2.5 else "❌")
                    )
                    count += 1
                except:
                    continue
        with open(self.prediction_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        return count, records

    def _save_model(self, ticker, model, scaler, last_loss, input_dim):
        """安全儲存模型"""
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "last_loss": last_loss,
                "input_dim": input_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "version": self.MODEL_VERSION,
            },
            path,
        )
