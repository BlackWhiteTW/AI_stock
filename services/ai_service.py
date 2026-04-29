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
from .ai_model_defs import LSTMModel, HybridTransLSTM
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

        # 🚀 新增：硬體設備自動偵測
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📡 AI 服務啟動，運算設備: {self.device}")

        print("📊 [1/4] 正在初始化市場情緒分析器...")
        self.sentiment_analyzer = MarketSentimentAnalyzer()

        # 🆕 v5.4 新增：周期性新聞收集器
        print("📊 [2/4] 正在初始化新聞收集器...")
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
        self.records_dir = "data/training_records" 
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
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)

        print("📊 [3/4] 正在初始化交易策略引擎...")
        self.strategy = StockStrategy()
        
        print("📊 [4/4] 正在掃描本地模型...")
        self._load_all_models()
        print(f"✨ AI 服務初始化完成 (共載入 {len(self.models)} 檔模型資訊)")

    def _save_training_record(self, ticker, loss, model_type="Individual"):
        """💾 保存訓練記錄到 JSON，供 UI 歷史分頁讀取"""
        try:
            record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "loss": float(loss),
                "model_type": model_type,
                "version": self.MODEL_VERSION
            }
            # 檔案名包含時間戳與代碼
            filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{ticker.replace('.', '_')}.json"
            file_path = os.path.join(self.records_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"⚠️ 無法保存訓練記錄: {e}")
            return False

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
            model_type = checkpoint.get("model_type", "LSTMModel") # 預設為舊版 LSTM

            # 🚀 根據類型實例化模型
            if model_type == "HybridTransLSTM":
                model = HybridTransLSTM(
                    input_dim=input_dim, 
                    hidden_dim=hidden_dim, 
                    num_layers=num_layers,
                    seq_len=self.lookback
                )
            else:
                model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1)
                
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(self.device) # 🚀 確保移至正確設備
            model.float()        # 🚀 確保權重為 Float32
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

        # 🚀 確保所有特徵欄位都存在，若缺失則補 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # 處理 Infinity 異常值
        data_clean = df[self.feature_columns].copy()
        
        # 1. 將無限大替換為 NaN，然後填充為 0
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.fillna(0)
        
        # 2. 截斷極端值 (例如大於 10 億或小於 -10 億的異常數據)
        data_clean = data_clean.clip(lower=-1e9, upper=1e9)

        # 執行切片
        data_clean = data_clean.iloc[60:].copy()
        if data_clean.empty:
            # 如果切片後沒數據，至少回傳原始結構填充 0
            data_clean = df[self.feature_columns].replace([np.inf, -np.inf], 0).fillna(0).copy()

        # 高波動股票對數處理
        price_cv = data_clean["Close"].std() / (data_clean["Close"].mean() + 1e-8)
        if price_cv > 0.6:
            for feat in ["Close", "Open", "High", "Low"]:
                # 確保 log1p 內的值為正數
                data_clean[feat] = np.log1p(data_clean[feat].clip(lower=0))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # 再次檢查是否存在 NaN (雖然前面已填 0，但 log1p 可能產生新 NaN)
        final_values = data_clean.values
        if not np.isfinite(final_values).all():
            final_values = np.nan_to_num(final_values, nan=0.0, posinf=0.0, neginf=0.0)
            
        scaled_data = scaler.fit_transform(final_values)

        X, y = [], []
        for i in range(len(scaled_data) - lookback - 1):
            X.append(scaled_data[i : i + lookback])
            y.append(scaled_data[i + lookback, 0])  # 預測目標：Close

        return np.array(X), np.array(y), scaler

    def train_deep_learning_model(
        self, ticker, df, progress_callback=None, use_transfer_learning=True
    ):
        """
        🔥 專業級旗艦訓練引擎 (升級為 HybridTransLSTM)
        包含 GPU 加速、SWA、重度方向性損失優化
        """
        if len(df) > 3000:
            df = df.iloc[-3000:].copy()
        if len(df) < 200:
            return "⚠️ 數據不足 (至少需 200 天)", 0

        # 特徵工程
        print(f"  📊 正在為 {ticker} 進行特徵工程...")
        df = tech_analysis.calculate_technical_indicators(
            df, ticker=ticker, include_financial=True
        )
        self.strategy.optimize_strategy_basic(ticker, df)

        X, y, scaler = self.prepare_lstm_data(df, self.lookback, ticker=ticker)

        # 數據分割 (使用 90/10 分割，保留更多訓練數據)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )
        
        # 🚀 修正：在最上層就強轉為 Tensor，避免後續 batch 處理索引問題與效能損耗
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # 🚀 升級：使用 HybridTransLSTM 旗艦架構
        input_dim = X.shape[2]
        hidden_dim = 256
        model = HybridTransLSTM(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=3, 
            seq_len=self.lookback
        ).to(self.device)
        model.float() # 🚀 強制確保權重為 Float32

        # 🚀 遷移學習：100% 匹配架構後才能發揮威力
        if use_transfer_learning and os.path.exists(self.base_model_path):
            try:
                base = torch.load(self.base_model_path, map_location=self.device, weights_only=False)
                if base.get("model_type") == 'HybridTransLSTM' and base["input_dim"] == input_dim:
                    model.load_state_dict(base["model_state_dict"], strict=False)
                    model.float() # 加載後再次確保
                    print(f"  ✓ 成功繼承旗艦級主模型權重")
            except Exception as e:
                print(f"  ⚠️ 權重繼承跳過: {e}")

        # 損失函數：Huber + 重度方向懲罰
        criterion_base = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

        best_val_loss = float("inf")
        start_time = time.time()
        epochs = 80 # 個股訓練不需要太多輪，避免過擬合

        print(f"  🚀 開始訓練 {ticker} (架構: HybridTransLSTM)")

        for epoch in range(epochs):
            model.train()
            # 隨機打亂 Batch
            perm = torch.randperm(len(X_train_t))
            train_loss = 0
            
            for i in range(0, len(X_train_t), 32):
                idx = perm[i:i+32]
                batch_X = X_train_t[idx]
                batch_y = y_train_t[idx]

                optimizer.zero_grad()

                outputs = model(batch_X)
                targets = batch_y.unsqueeze(1)

                # 1. 基礎誤差
                loss_mse = criterion_base(outputs, targets)

                # 2. 🚀 強化方向性懲罰 (提高權重到 2.0)
                # 💡 安全索引：確保在 3D 張量下才執行序列取值
                if batch_X.dim() == 3 and batch_X.size(1) > 0:
                    try:
                        # 取得最後一天的收盤價作為基準
                        last_closes = batch_X[:, -1, 0].unsqueeze(1)
                        pred_diff = outputs - last_closes
                        real_diff = targets - last_closes
                        direction_mismatch = (pred_diff * real_diff) < 0
                        directional_loss = torch.mean(direction_mismatch.float() * (outputs - targets) ** 2) * 2.0
                    except Exception as e:
                        # 發生索引錯誤時安全退回
                        print(f"⚠️ 方向損失計算失敗: {e}")
                        directional_loss = 0.0
                else:
                    directional_loss = 0.0

                loss = loss_mse + directional_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # 驗證
            if epoch % 5 == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    v_y = y_val_t.unsqueeze(1)
                    out = model(X_val_t)
                    val_loss = criterion_base(out, v_y).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model(ticker, model, scaler, val_loss, input_dim)
                    # 🚀 新增：保存訓練記錄
                    self._save_training_record(ticker, val_loss, "Individual")

                if progress_callback:
                    progress_callback(epoch + 1, epochs, val_loss, time.time() - start_time)

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

        # 2. 篩選代表性標的與樣本 (🚀 增強版增加標的數量)
        tickers = full_df['Ticker'].unique()
        sample_count = 200 if enhanced else 100
        selected_tickers = np.random.choice(tickers, min(sample_count, len(tickers)), replace=False)
        
        all_X, all_y = [], []
        for i, ticker in enumerate(selected_tickers):
            df_ticker = full_df[full_df['Ticker'] == ticker].copy()
            if len(df_ticker) < 500: continue
            
            # 計算必要指標
            df_ticker = tech_analysis.calculate_technical_indicators(df_ticker, ticker=ticker)
            X, y, _ = self.prepare_lstm_data(df_ticker, self.lookback, ticker=ticker)
            
            if X is not None and len(X) > 0:
                # 強制清洗：確保數據中沒有 NaN 或 Inf
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 抽樣以維持 VRAM 穩定性
                indices = np.random.choice(len(X), min(200, len(X)), replace=False)
                all_X.append(X[indices]); all_y.append(y[indices])
            
            if progress_callback:
                progress_callback(i + 1, len(selected_tickers), -1, 0, ticker=ticker)

        if not all_X:
            return "❌ 無法生成足夠的訓練樣本，請檢查數據源。"

        # 合併數據並進行最後一次全局安全檢查
        try:
            # 修正：使用 concatenate 確保樣本沿著第 0 維度串接，避免維度錯誤
            X_combined = np.nan_to_num(np.concatenate(all_X, axis=0), nan=0.0).astype(np.float32)
            y_combined = np.nan_to_num(np.concatenate(all_y, axis=0), nan=0.0).astype(np.float32)
            
            print(f"📊 訓練數據準備就緒: X={X_combined.shape}, y={y_combined.shape}, 正在遷移至 {self.device}...")
            
            # 使用 torch.tensor 並強制 contiguous
            X_train = torch.tensor(X_combined, dtype=torch.float32).contiguous().to(self.device)
            y_train = torch.tensor(y_combined, dtype=torch.float32).contiguous().to(self.device).unsqueeze(-1)
            
        except Exception as e:
            print(f"⚠️ GPU 數據遷移失敗 ({e})，嘗試切換至 CPU 模式...")
            self.device = torch.device("cpu")
            X_train = torch.tensor(X_combined, dtype=torch.float32)
            y_train = torch.tensor(y_combined, dtype=torch.float32).unsqueeze(-1)

        # 3. 初始化旗艦級混合模型 (🚀 增強版增加 hidden_dim)
        hidden_dim = 256 if enhanced else 128
        try:
            model = HybridTransLSTM(
                input_dim=len(self.feature_columns),
                hidden_dim=hidden_dim,
                num_layers=3,
                seq_len=self.lookback
            ).to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"❌ 模型載入 GPU 失敗，強退至 CPU: {e}")
                self.device = torch.device("cpu")
                model = HybridTransLSTM(
                    input_dim=len(self.feature_columns),
                    hidden_dim=128,
                    num_layers=3,
                    seq_len=self.lookback
                ).to(self.device)
            else:
                raise e
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        
        # 4. 旗艦級訓練流程
        epochs = 100 if enhanced else 50
        batch_size = 256
        
        print(f"🏃 開始執行主模型訓練 (Device: {self.device})...")
        
        try:
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(len(X_train))
                epoch_loss = 0
                for k in range(0, len(X_train), batch_size):
                    idx = perm[k:k+batch_size]
                    bx, by = X_train[idx], y_train[idx]
                    
                    # 🚀 數值穩定性：再次確保輸入不包含極端值
                    bx = torch.clamp(bx, -1e6, 1e6)
                    
                    optimizer.zero_grad()
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                    
                    if torch.isnan(loss):
                        print(f"⚠️ 偵測到 NaN Loss，跳過此 Batch")
                        continue
                        
                    loss.backward()
                    
                    # 🚀 關鍵補強：梯度截斷，防止 Transformer 爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / (len(X_train) / batch_size)
                if progress_callback: progress_callback(epoch + 1, epochs, avg_loss, 0)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        except RuntimeError as e:
            if "CUDA" in str(e) or "device-side assert" in str(e):
                print(f"🚨 訓練中途發生 CUDA 崩潰: {e}")
                print("🔄 正在強制降級至 CPU 並嘗試修復...")
                # 降級邏輯
                self.device = torch.device("cpu")
                model = model.to("cpu")
                X_train = X_train.to("cpu")
                y_train = y_train.to("cpu")
                # 遞迴呼叫自己（但現在 device 已經是 cpu，會安全跑完）
                return self.train_base_model(stock_list, progress_callback, enhanced)
            else:
                raise e
                
        # 5. 保存旗艦級模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': self.feature_columns,
            'input_dim': len(self.feature_columns),
            'hidden_dim': getattr(model, 'hidden_dim', 256),
            'num_layers': getattr(model, 'num_layers', 3),
            'model_type': 'HybridTransLSTM',
            'version': '2026.旗艦版'
        }, self.base_model_path)
        
        # 🚀 新增：保存主模型訓練記錄
        self._save_training_record("BASE_MODEL", avg_loss, "Flagship_Base")
        
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
            X = torch.from_numpy(np.array([data_scaled])).float().to(self.device)

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
            
            # 🆕 新增：成交量異常檢測 (轉換為 0/1 的 float)
            df["Volume_Anomaly"] = self.sentiment_analyzer.detect_volume_anomaly(df).astype(float)
            
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

    def run_morning_predictions(self, force=False):
        """開盤前批量預測紀錄與新聞收集 (增量存檔版)"""
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        
        # 🆕 週末檢查：避免參數失準
        if not force and now.weekday() >= 5: # 5: Saturday, 6: Sunday
            print(f"📅 [Morning Job] 今日為週末 ({now.strftime('%A')})，跳過自動預測任務以維持參數精準度。")
            return 0

        # 🆕 v5.4: 在開盤前自動執行每日新聞收集
        if self.news_collector:
            try:
                print("📰 [Morning Job] 正在執行每日新聞收集...")
                self.news_collector.collect_daily_news()
            except Exception as e:
                print(f"⚠️ 每日新聞收集失敗: {e}")

        ticker_list = list(self.models.keys())
        
        # 🚀 修正：改為逐個預測並立刻存檔，避免中斷導致全部遺失
        count = 0
        for ticker in ticker_list:
            try:
                print(f"🔮 [Morning Job] 正在預測 {ticker} ... ({count+1}/{len(ticker_list)})")
                _, hist, _, _ = get_stock_data(ticker)
                
                if hist.empty or len(hist) < self.lookback:
                    continue
                    
                # 🛠️ 數據日期檢查：確保 base_price 是最新的 (通常是昨天或上一個交易日)
                # 🆕 修正：如果最後一筆是今天且成交量為 0，則基準價應取前一筆
                last_row = hist.iloc[-1]
                if last_row.name.date() == now.date() and last_row["Volume"] == 0:
                    base_row = hist.iloc[-2]
                else:
                    base_row = last_row

                last_data_date = base_row.name.strftime("%Y-%m-%d")
                base = float(base_row["Close"])
                open_p = float(base_row["Open"])
                
                pred = self.predict_next_price(ticker, hist)
                
                if pred:
                    new_rec = {
                        "date": today_str,
                        "data_last_date": last_data_date, # 紀錄數據源的最後日期
                        "ticker": ticker,
                        "base_price": base,       # 昨收價 (或數據最後一筆收盤)
                        "open_price": open_p,     # 數據最後一筆開盤
                        "predicted_price": float(pred),
                        "expected_change_pct": round((pred - base) / base * 100, 2),
                        "actual_price": None,
                        "diff_pct": None,         # 🆕 初始化誤差百分比
                        "accuracy": "等待收盤",
                    }
                    
                    # 讀取現有紀錄
                    all_recs = []
                    if os.path.exists(self.prediction_file):
                        try:
                            with open(self.prediction_file, "r", encoding="utf-8") as f:
                                all_recs = json.load(f)
                        except: pass
                    
                    # 移除該股票今日的舊預測，加入新的 (Upsert 邏輯)
                    all_recs = [r for r in all_recs if not (r["date"] == today_str and r["ticker"] == ticker)]
                    all_recs.append(new_rec)
                    
                    # 立刻存檔
                    with open(self.prediction_file, "w", encoding="utf-8") as f:
                        json.dump(all_recs, f, ensure_ascii=False, indent=4)
                    
                    count += 1
            except Exception as e:
                print(f"⚠️ 預測 {ticker} 失敗: {e}")
                continue

        return count

    def validate_afternoon_results(self, force=False):
        """收盤後驗證準確度 (增量存檔版)"""
        if not os.path.exists(self.prediction_file):
            return 0, []
            
        with open(self.prediction_file, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        count = 0
        
        for r in records:
            # 🆕 修正：不僅針對今日，只要是「等待收盤」或「之前驗證失敗/疑似錯誤」的都可以嘗試驗證
            should_validate = (
                not r.get("actual_price") or 
                r.get("accuracy") == "等待收盤" or 
                r.get("accuracy") in ["❌", "??", "error"] or
                r.get("diff_pct") is None
            )
            
            if should_validate:
                try:
                    ticker = r["ticker"]
                    target_date = r["date"] # 預測目標日期
                    
                    print(f"✅ [Afternoon Job] 正在嘗試驗證 {ticker} (目標日期: {target_date})...")
                    # 🚀 強制不使用快取，獲取最新數據
                    _, hist, _, _ = get_stock_data(ticker, use_cache=False)
                    
                    if not hist.empty:
                        # 確保索引是日期時間格式且無時區
                        if hist.index.tz is not None:
                            hist.index = hist.index.tz_localize(None)
                            
                        last_data_date = hist.index[-1].strftime("%Y-%m-%d")
                        
                        # 🛠️ 修改：如果 last_data_date == target_date，即視為可驗證 (即時價格)
                        if not force and last_data_date < target_date:
                            print(f"⏳ [Afternoon Job] {ticker} 數據尚未更新 (最新為 {last_data_date}，目標為 {target_date})，跳過。")
                            continue
                            
                        # 找到目標日期的收盤價，如果找不到則用最後一筆
                        try:
                            # 嘗試精確匹配日期
                            target_row = hist.loc[hist.index.strftime("%Y-%m-%d") == target_date]
                            if not target_row.empty:
                                # 🆕 額外檢查：成交量需大於 0 (排除盤前占位數據)
                                if target_row["Volume"].iloc[0] == 0 and not force:
                                    print(f"⏳ [Afternoon Job] {ticker} 目標日期數據成交量為 0，視為尚未開盤或停盤，跳過。")
                                    continue
                                    
                                actual = float(target_row["Close"].iloc[0])
                                actual_date = target_date
                            else:
                                # 若無精確日期，且數據最新日期已超過目標日，才取最後一筆 (例如預測在週末)
                                if last_data_date > target_date:
                                    actual = float(hist["Close"].iloc[-1])
                                    actual_date = last_data_date
                                else:
                                    # 數據還沒追上目標日期，繼續等待
                                    continue
                        except Exception as e:
                            print(f"⚠️ 提取價格時發生錯誤: {e}")
                            continue

                        r["actual_price"] = actual
                        r["actual_date"] = actual_date
                        
                        # 計算準確度
                        diff = abs(r["predicted_price"] - actual) / actual * 100
                        r["diff_pct"] = round(diff, 2)
                        r["accuracy"] = (
                            "⭐⭐⭐" if diff < 1.0 else ("⭐" if diff < 2.5 else "❌")
                        )
                        count += 1
                        
                        # 立刻存檔 (增量保存)
                        with open(self.prediction_file, "w", encoding="utf-8") as f:
                            json.dump(records, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"⚠️ 驗證 {r.get('ticker')} 失敗: {e}")
                    continue
                    
        return count, records

    def _save_model(self, ticker, model, scaler, last_loss, input_dim):
        """安全儲存模型"""
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        # 🚀 紀錄模型類型
        model_type = "HybridTransLSTM" if isinstance(model, HybridTransLSTM) else "LSTMModel"
        
        # 🚀 使用 getattr 安全獲取參數，避免環境快取舊定義導致報錯
        h_dim = getattr(model, 'hidden_dim', 256)
        n_layers = getattr(model, 'num_layers', 3)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "last_loss": last_loss,
                "input_dim": input_dim,
                "hidden_dim": h_dim,
                "num_layers": n_layers,
                "model_type": model_type,
                "version": self.MODEL_VERSION,
            },
            path,
        )
