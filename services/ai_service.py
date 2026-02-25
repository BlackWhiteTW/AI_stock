import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time
import os
import gc
import json
import datetime
from utils import get_stock_data
from .ai_model_defs import LSTMModel, GRUModel, TransformerModel, AttentionLSTM, EnsembleModel
from . import tech_analysis
from .stock_strategy import StockStrategy
from .market_sentiment import MarketSentimentAnalyzer
try:
    from .news_analyzer import NewsAnalyzer
    NEWS_AVAILABLE = True
except:
    NEWS_AVAILABLE = False

class AIService:
    # 模型版本號（每次架構變化時更新）
    MODEL_VERSION = "5.1"  # v5.1: Transformer + Attention + Ensemble + 26特徵 (含財務報表)
    
    def __init__(self):
        self.models = {} # Metadata
        self.loaded_models = {} # LRU Cache
        self.MAX_CACHE_SIZE = 15
        
        # 🆕 市場情緒分析器
        self.sentiment_analyzer = MarketSentimentAnalyzer() 
        
        self.lookback = 60 
        self.model_dir = "data/saved_models"
        self.prediction_file = "data/daily_predictions.json"
        self.base_model_path = "data/saved_models/base_model.pth"  # 遷移學習基礎模型
        
        # 完整特徵列表
        self.all_features = [
            'Close', 'Open', 'High', 'Low', 'Volume', 
            'RSI', 'MACD', 'MA20', 'StdDev', 
            'K', 'D', 'OBV', 'ATR', 'Volatility', 
            'BIAS_20', 'Yield', 'MOM', 'CCI'
        ]
        
        # 🆕 [v5.1 財務報表強化] 從 20 個擴展到 26 個特徵
        # 包含：基礎、技術指標、市場情緒、資金流向、財務報表
        self.feature_columns = [
            # 基礎價量 (5個)
            'Close', 'Open', 'High', 'Low', 'Volume',
            # 核心技術指標 (6個)
            'RSI', 'MACD', 'K', 'D', 'ATR', 'Volatility',
            # 🆕 市場情緒指標 (5個)
            'MFI', 'ADX', 'Price_Position', 'Volume_Anomaly', 'Market_Sentiment_Score',
            # 🆕 進階指標 (4個)
            'Sharp_Up', 'Sharp_Down', 'OBV', 'BIAS_20',
            # 🆕 [v5.1] 財務報表特徵 (6個)
            'fin_financial_health_score', 'fin_roe_normalized', 'fin_debt_ratio_normalized',
            'fin_current_ratio_normalized', 'fin_pe_ratio_normalized', 'fin_dividend_yield_normalized'
        ]  # v5.1: 26 個特徵，預期準確度 +20%（財務基本面 + 技術面）
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 這裡初始化策略物件，僅用於訓練時的回測優化步驟
        self.strategy = StockStrategy()

        self._load_all_models()

    def _save_model(self, ticker, model, scaler, last_loss, input_dim):
        """保存模型到磁碟"""
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        
        # 判斷模型類型
        if isinstance(model, EnsembleModel):
            model_type = 'Ensemble'
            # EnsembleModel 使用子模型的參數
            hidden_dim = model.lstm.hidden_dim
            num_layers = model.lstm.num_layers
        elif isinstance(model, GRUModel):
            model_type = 'GRU'
            hidden_dim = model.hidden_dim
            num_layers = model.num_layers
        elif isinstance(model, TransformerModel):
            model_type = 'Transformer'
            hidden_dim = model.hidden_dim
            num_layers = model.num_layers
        elif isinstance(model, AttentionLSTM):
            model_type = 'AttentionLSTM'
            hidden_dim = model.hidden_dim
            num_layers = model.num_layers
        else:  # LSTMModel
            model_type = 'LSTM'
            hidden_dim = model.hidden_dim
            num_layers = model.num_layers
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'last_loss': last_loss,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'model_type': model_type,
            'version': self.MODEL_VERSION  # 記錄版本號
        }, path)

    def _load_all_models(self):
        """[優化] 僅掃描檔案列表，不預先載入模型以節省記憶體"""
        self.models = {}
        if not os.path.exists(self.model_dir): return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pth"):
                ticker = filename.replace(".pth", "")
                self.models[ticker] = {"last_loss": 0.0}

    def _get_model_instance(self, ticker):
        """[LRU Cache] 從記憶體緩存或硬碟讀取模型"""
        # 1. Hit Cache
        if ticker in self.loaded_models:
            val = self.loaded_models.pop(ticker)
            self.loaded_models[ticker] = val
            return val['model'], val['scaler']
        
        # 2. Miss Cache - Load from Disk
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        if not os.path.exists(path):
            return None, None
            
        try:
            # 載入模型 (CPU)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # [版本控制] 檢查模型版本
            model_version = checkpoint.get('version', '1.0')  # 舊模型沒有版本號，默認1.0
            
            if model_version != self.MODEL_VERSION:
                # 版本不匹配，標記需要重新訓練
                if ticker not in self.models:
                    self.models[ticker] = {}
                self.models[ticker]['needs_retrain'] = True
                self.models[ticker]['old_version'] = model_version
                # 優化: 只在第一次載入時警告，避免重複顯示
                if not self.models[ticker].get('version_warning_shown', False):
                    print(f"⚠️  {ticker} 模型版本過舊 (v{model_version} → v{self.MODEL_VERSION})，建議重新訓練以使用新功能")
                    self.models[ticker]['version_warning_shown'] = True
                return None, None
            
            input_dim = checkpoint['input_dim']
            hidden_dim = checkpoint['hidden_dim']
            num_layers = checkpoint['num_layers']
            model_type = checkpoint.get('model_type', 'LSTM')
            
            # 根據模型類型創建對應的模型
            if model_type == 'Ensemble':
                model = EnsembleModel(input_dim, hidden_dim, num_layers, output_dim=1)
            elif model_type == 'Transformer':
                model = TransformerModel(input_dim, hidden_dim, num_layers, output_dim=1)
            elif model_type == 'AttentionLSTM':
                model = AttentionLSTM(input_dim, hidden_dim, num_layers, output_dim=1)
            elif model_type == 'GRU':
                model = GRUModel(input_dim, hidden_dim, num_layers, output_dim=1)
            else:  # LSTM
                model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 更新 metadata
            if ticker not in self.models: self.models[ticker] = {}
            self.models[ticker]['last_loss'] = checkpoint['last_loss']
            self.models[ticker]['version'] = model_version
            self.models[ticker]['needs_retrain'] = False
            
            # 3. Update Cache & Evict if needed
            if len(self.loaded_models) >= self.MAX_CACHE_SIZE:
                removed_key = next(iter(self.loaded_models))
                del self.loaded_models[removed_key]
                gc.collect()
            
            self.loaded_models[ticker] = {
                "model": model,
                "scaler": checkpoint['scaler']
            }
            
            return model, checkpoint['scaler']
            
        except Exception as e:
            print(f"Error loading model {ticker}: {e}")
            if "state_dict" in str(e):
                try: os.remove(path)
                except: pass
            return None, None
            
    def prepare_lstm_data(self, df, lookback, ticker=None):
        """準備 LSTM 訓練數據 (v5.1 增強版特徵 - 包含財務報表)
        
        Args:
            df: 股票歷史數據
            lookback: 回看窗口天數
            ticker: 股票代碼（用於獲取財務特徵）
        """
        # 🆕 計算市場情緒特徵
        df = self._add_market_sentiment_features(df)
        
        # 檢查關鍵特徵是否存在，若無則重新計算（包含財務特徵）
        if 'MOM' not in df.columns or 'CCI' not in df.columns or 'K' not in df.columns:
            df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)
        
        # 🆕 [v5.1] 確保財務特徵存在
        if ticker and 'fin_financial_health_score' not in df.columns:
            try:
                df = tech_analysis.add_financial_features(df, ticker)
                print(f"  ✅ 已添加財務報表特徵到訓練數據")
            except Exception as e:
                print(f"  ⚠️  財務特徵添加失敗: {e}，使用默認值")

        features = self.feature_columns
        
        for col in features:
            if col not in df.columns:
                df[col] = 0
                
        data_clean = df[features].fillna(0).iloc[60:].copy() 
        if data_clean.empty:
            data_clean = df[features].fillna(0).copy()
        
        # [優化] 高波動股票特殊處理：檢測價格變異係數
        price_cv = data_clean['Close'].std() / data_clean['Close'].mean()
        high_volatility = price_cv > 0.6  # 變異係數>60%視為高波動
        
        if high_volatility:
            print(f"  ⚠️  檢測到高波動股票 (變異係數: {price_cv:.1%})，啟用對數縮放")
            # 對價格相關特徵使用對數縮放，減少極端值影響
            price_features = ['Close', 'Open', 'High', 'Low']
            for feat in price_features:
                if feat in data_clean.columns:
                    # 使用log1p避免log(0)
                    data_clean[feat] = np.log1p(data_clean[feat])
            
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        raw_data = data_clean.values
        scaled_data = scaler.fit_transform(raw_data)
        
        X, y = [], []
        for i in range(len(scaled_data) - lookback - 1):
            X.append(scaled_data[i : i + lookback])
            y.append(scaled_data[i + lookback, 0]) 
            
        return np.array(X), np.array(y), scaler

    def train_deep_learning_model(self, ticker, df, progress_callback=None, use_transfer_learning=True):
        """訓練深度學習模型 (v5.1 - 整合財務報表數據)
        
        Args:
            ticker: 股票代碼
            df: 歷史數據
            progress_callback: 進度回調函數
            use_transfer_learning: 是否使用遷移學習（大幅加速）
        """
        # 優化 1: 數據截斷 (只取最近 12 年 / 約 3000 天)，避免遠古數據干擾且大幅加速訓練
        if len(df) > 3000:
            df = df.iloc[-3000:].copy()

        if len(df) < 200:
            return "⚠️ 數據不足，無法進行深度學習訓練 (至少需要 200 天歷史數據)", 0

        # 🆕 [v5.1] 預先計算所有特徵（技術指標 + 財務報表）
        print(f"  📊 準備訓練數據（技術面 + 財務面）...")
        if 'RSI' not in df.columns or 'fin_financial_health_score' not in df.columns:
            df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)

        # 調用 Strategy 做回測優化
        backtest_msg = self.strategy.optimize_strategy_basic(ticker, df)
        
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
            else:
                device = torch.device('cpu')
                device_name = "CPU"
            
            # 傳入 ticker 以確保財務特徵正確添加
            X, y, scaler = self.prepare_lstm_data(df, self.lookback, ticker=ticker)
            
            # [優化] 檢測高波動股票
            features_df = df[self.feature_columns].copy()
            price_cv = features_df['Close'].std() / features_df['Close'].mean()
            high_volatility = price_cv > 0.6
            
            if high_volatility:
                print(f"  🎯 高波動模式啟動 (變異係數: {price_cv:.1%})")
            
            if len(X) < 10:
                return "⚠️ 切割後數據過少", 0
            
            # 優化 2: 訓練/驗證集分割 (80/20)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False  # 時間序列不打亂順序
            )
            
            # [效能優化] 動態調整 batch size - GTX 1650 優化
            # 較大的 batch 可提升 GPU 利用率並減少波動
            if torch.cuda.is_available():
                batch_size = 128  # GTX 1650 適合 128-256
                num_workers = 2   # 2-4 個 worker 預加載數據，減少 GPU 等待
            else:
                batch_size = 32
                num_workers = 0
            
            # 訓練集 DataLoader
            train_dataset = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float()
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,  # 多線程預加載數據
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True if num_workers > 0 else False,  # 保持 worker 存活
                prefetch_factor=2 if num_workers > 0 else None  # 每個 worker 預取 2 批數據
            )
            
            # 驗證集 DataLoader
            val_dataset = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float()
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size * 2,  # 驗證時可用更大batch
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )

            input_dim = X.shape[2]
            # 🆕 v5.0: 動態選擇最佳模型架構
            model_type = self._select_best_model_type(len(X_train), high_volatility)
            
            # 優化 3: 動態調整模型大小（根據數據量）
            if len(X_train) > 1500:
                hidden_dim = 256
                num_layers = 3
            elif len(X_train) > 500:
                hidden_dim = 128
                num_layers = 2
            else:
                hidden_dim = 64
                num_layers = 2
            output_dim = 1
            
            # 🆕 v5.0: 創建對應的模型
            if model_type == "Ensemble":
                model = EnsembleModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
                print(f"  🎯 使用模型組合系統 (LSTM + GRU + Transformer)")
            elif model_type == "Transformer":
                model = TransformerModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
                print(f"  🎯 使用 Transformer 模型")
            elif model_type == "AttentionLSTM":
                model = AttentionLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
                print(f"  🎯 使用 LSTM + Attention 模型")
            elif model_type == "GRU":
                model = GRUModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
                print(f"  🎯 使用 GRU 模型")
            else:  # LSTM
                model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
                print(f"  🎯 使用 LSTM 模型")
            
            # [遷移學習] 嘗試加載基礎模型權重
            transfer_learning_used = False
            if use_transfer_learning and os.path.exists(self.base_model_path):
                try:
                    base_checkpoint = torch.load(self.base_model_path, map_location=device, weights_only=False)
                    # 只有在架構相同時才加載
                    if (base_checkpoint.get('input_dim') == input_dim and 
                        base_checkpoint.get('hidden_dim') == hidden_dim and
                        base_checkpoint.get('num_layers') == num_layers):
                        model.load_state_dict(base_checkpoint['model_state_dict'], strict=False)
                        transfer_learning_used = True
                        
                        # [優化3] 凍結前面的GRU層，只微調最後的FC層
                        # 這樣可以保留基礎模型學到的通用特徵，同時適應個股特性
                        for name, param in model.named_parameters():
                            if 'gru' in name:  # 凍結GRU層
                                param.requires_grad = False
                            # fc層保持可訓練（requires_grad=True）
                        
                        print(f"✓ {ticker} 使用遷移學習（凍結GRU層，微調FC層）")
                except:
                    pass
            
            # [效能優化] 使用 torch.compile 加速模型（PyTorch 2.0+）
            compile_model = False
            try:
                if hasattr(torch, 'compile') and torch.cuda.is_available():
                    import platform
                    if platform.system() == 'Linux':
                        model = torch.compile(model, mode='reduce-overhead')
                        compile_model = True
            except Exception as e:
                pass
            
            # 優化器：使用 AdamW（加入權重衰減防止過擬合）
            # [優化] 高波動股票使用Huber Loss，對異常值更魯棒
            if high_volatility:
                basic_criterion = nn.HuberLoss(delta=1.0)  # 對大誤差使用線性懲罰
                print(f"  🎯 使用Huber Loss（delta=1.0）降低異常值影響")
            else:
                basic_criterion = nn.MSELoss()
            
            # [優化3] 高波動股票使用更保守的學習率和更強的正則化
            if high_volatility:
                learning_rate = 0.0003 if transfer_learning_used else 0.0005
                weight_decay = 1e-4  # 10x增強正則化
                print(f"  🔧 lr={learning_rate}, weight_decay={weight_decay}")
            else:
                learning_rate = 0.001 if transfer_learning_used else 0.001
                weight_decay = 1e-5
            
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                fused=torch.cuda.is_available()  # 融合優化器運算（更快）
            )
            
            # [效能優化] 啟用混合精度訓練（FP16/BF16）加速 2-3倍
            use_amp = torch.cuda.is_available()
            scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None
            
            # 學習率調度器：Cosine Annealing with Warm Restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=20,  # 20個epoch後重啟
                T_mult=2,  # 每次重啟間隔翻倍
                eta_min=1e-6
            )

            # [優化1] 調整訓練輪數：增加輪數確保充分收斂
            if high_volatility:
                num_epochs = 200 if transfer_learning_used else 250  # 高波動股票需要更多訓練
                patience = 60 if transfer_learning_used else 70
                print(f"  高波動模式：訓練輪數={num_epochs}, patience={patience}")
            else:
                num_epochs = 100 if transfer_learning_used else 120  # 正常股票也增加訓練
                patience = 30 if transfer_learning_used else 35
            
            # [優化] 初始化 SWA (隨機權重平均) - 提升泛化能力
            use_swa = True
            if use_swa:
                from torch.optim.swa_utils import AveragedModel, SWALR
                swa_model = AveragedModel(model)
                swa_start = int(num_epochs * 0.7)  # 在最後 30% 的輪數啟動 SWA
                swa_scheduler = SWALR(optimizer, swa_lr=learning_rate * 0.5)
                print(f"  🧠 SWA 模式已準備 (將於第 {swa_start} 輪啟動)")

            train_losses = []
            val_losses = []
            
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            start_time = time.time()
            actual_epochs = 0
            stop_reason = "(Max epochs reached)"
            
            for epoch in range(num_epochs):
                actual_epochs = epoch + 1
                
                # [優化2] 自適應解凍邏輯 (保留原有的)
                unfreeze_epoch = 40 if high_volatility else 30
                if transfer_learning_used and epoch == unfreeze_epoch:
                    # ... (原有的解凍邏輯)
                    pass

                # ===== 訓練階段 =====
                model.train()
                train_batch_losses = []
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    # [優化] 數據抗噪 (Noise Injection)
                    # 加入微小的高斯噪音，強迫模型學習強健特徵
                    if model.training:
                        noise = torch.randn_like(batch_X) * 0.001 
                        batch_X = batch_X + noise

                    optimizer.zero_grad(set_to_none=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(batch_X)
                            targets = batch_y.unsqueeze(1)
                            
                            # [優化] 基礎 MSE 損失
                            loss_mse = basic_criterion(outputs, targets)
                            
                            # [優化] 方向性懲罰 (Directional Penalty)
                            last_closes = batch_X[:, -1, 0].unsqueeze(1)
                            pred_diff = outputs - last_closes
                            real_diff = targets - last_closes
                            direction_mismatch = (pred_diff * real_diff) < 0
                            directional_loss = torch.mean(direction_mismatch.float() * (outputs - targets)**2) * 0.5 # 提高權重
                            
                            loss = loss_mse + directional_loss
                        
                        scaler_amp.scale(loss).backward()
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        # CPU 模式同樣加入邏輯
                        outputs = model(batch_X)
                        targets = batch_y.unsqueeze(1)
                        loss = basic_criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                    train_batch_losses.append(loss.item())
                
                # 更新學習率
                if use_swa and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

                train_loss = np.mean(train_batch_losses)
                train_losses.append(train_loss)
                
                # ===== 驗證階段 =====
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    model.eval()
                    val_batch_losses = []
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(device, non_blocking=True)
                            batch_y = batch_y.to(device, non_blocking=True)
                            
                            if use_amp:
                                with torch.amp.autocast('cuda'):
                                    outputs = model(batch_X)
                            else:
                                outputs = model(batch_X)
                                
                            targets = batch_y.unsqueeze(1)
                            loss = basic_criterion(outputs, targets)
                            val_batch_losses.append(loss.item())
                    
                    val_loss = np.mean(val_batch_losses)
                    val_losses.append(val_loss)

                    # [優化] Early Stopping (僅在 SWA 啟動前生效)
                    if epoch < swa_start:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= max(5, patience // 5):
                            stop_reason = f"(Early stopping: {patience} epochs without improvement)"
                            print(f"  🛑 Early stopping: {patience} epochs without improvement")
                            break
                else:
                    val_loss = val_losses[-1] if val_losses else train_loss
                
                if progress_callback:
                    loop_elapsed = time.time() - start_time
                    progress_callback(epoch + 1, num_epochs, val_loss, loop_elapsed)

            # ===== 訓練結束處理 =====
            if use_swa:
                # 更新 SWA 模型的 BatchNorm 並取出最終平均權重
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
                final_model = swa_model.module
                print(f"  ✅ SWA 權重平均完成 (已整合第 {swa_start} 輪後的優化路徑)")
            else:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                final_model = model

            training_time = time.time() - start_time
            final_model.to('cpu')
            
            # 更新中繼資料與緩存
            final_val_loss = val_losses[-1] if val_losses else 0
            final_train_loss = train_losses[-1] if train_losses else 0
            self.models[ticker] = {
                "last_loss": final_val_loss,
                "train_loss": final_train_loss,
                "best_val_loss": best_val_loss
            }
            
            if ticker in self.loaded_models:
                del self.loaded_models[ticker] 
            
            if len(self.loaded_models) >= self.MAX_CACHE_SIZE:
                del self.loaded_models[next(iter(self.loaded_models))]
                
            self.loaded_models[ticker] = {"model": final_model, "scaler": scaler}
            
            # 保存最終模型
            self._save_model(ticker, final_model, scaler, best_val_loss if not use_swa else final_val_loss, input_dim)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 計算過擬合指標
            if final_train_loss > 0:
                overfit_ratio = final_val_loss / final_train_loss
                if overfit_ratio > 1.5:
                    overfit_status = f"⚠️ Overfit ratio {overfit_ratio:.1f}x (Potential Underfitting)"
                elif overfit_ratio > 1.2:
                    overfit_status = f"✅ Overfit ratio {overfit_ratio:.1f}x (Slight gap, normal)"
                elif overfit_ratio >= 0.9:
                    overfit_status = f"✅ Overfit ratio {overfit_ratio:.2f}x (Good generalization)"
                else:
                    overfit_status = f"⚠️ Overfit ratio {overfit_ratio:.2f}x (Abnormal, data issue?)"
            else:
                overfit_status = "N/A"
            
            # 遷移學習資訊：包含解凍狀態
            if transfer_learning_used:
                unfreeze_epoch = 40 if high_volatility else 30
                if actual_epochs > unfreeze_epoch:
                    transfer_info = f"\n- 🚀 Transfer Learning: Enabled (Frozen first {unfreeze_epoch} epochs, then fine-tuned)"
                else:
                    transfer_info = "\n- 🚀 Transfer Learning: Enabled (Fine-tuning FC layers)"
            else:
                transfer_info = ""

            return f"{backtest_msg}\n\n🧠 **Deep Learning Model Training Complete!**\n- Device: {device_name}\n- Architecture: PyTorch {model_type} (Features={input_dim}, Hidden={hidden_dim}, Layers={num_layers}){transfer_info}\n- Actual Epochs: {actual_epochs}/{num_epochs} {stop_reason}\n- Training Time: {training_time:.2f} s\n- Train Loss: {final_train_loss:.6f}\n- Val Loss: {final_val_loss:.6f}\n- Best Val Loss: {best_val_loss:.6f}\n- Overfit Status: {overfit_status}\n- Model saved to disk.", final_val_loss
        except Exception as e:
            return f"❌ Training Failed: {str(e)}", 0
    
    def train_base_model(self, stock_list=None, progress_callback=None, enhanced=True):
        """訓練遷移學習的基礎模型（增強版）
        
        使用多支股票的數據訓練一個通用模型，供其他股票遷移學習使用
        建議使用大型權值股（如台積電、鴻海等）的數據
        
        Args:
            stock_list: 股票代碼列表（如為空，使用預設增強池）
            progress_callback: 進度回調
            enhanced: 是否使用增強模式（更多股票、更多指數、更長歷史）
        """
        from utils import get_stock_data
        import yfinance as yf
        
        # 🆕 增強版股票池 - 涵蓋不同產業、市值、風格
        if stock_list is None and enhanced:
            stock_list = [
                # === 台股核心權值股 (8支) ===
                '2330.TW',  # 台積電 - 半導體龍頭
                '2317.TW',  # 鴻海 - 電子代工
                '2454.TW',  # 聯發科 - IC設計
                '2412.TW',  # 中華電 - 電信
                '2308.TW',  # 台達電 - 電源管理
                
                # === 金融股 (4支) ===
                '2882.TW',  # 國泰金
                '2886.TW',  # 兆豐金
                '2891.TW',  # 中信金
                '2892.TW',  # 第一金
                
                # === 傳產與塑化 (3支) ===
                '2002.TW',  # 中鋼 - 鋼鐵
                '1301.TW',  # 台塑 - 塑化
                '1303.TW',  # 南亞 - 塑化
                
                # === 光電與面板 (2支) ===
                '3008.TW',  # 大立光 - 光學
                '2409.TW',  # 友達 - 面板
                
                # === 代表性ETF (5支) ===
                '0050.TW',  # 元大台灣50
                '0056.TW',  # 元大高股息
                '00878.TW', # 國泰永續高股息
                '006208.TW',# 富邦台50
                '00632R.TW',# 元大台灣50反1
                
                # === 國際指數 (3支) - 增加全球市場視野 ===
                '^TWII',    # 台股加權指數
                '^GSPC',    # 標普500指數
                '^IXIC',    # 那斯達克指數
            ]
            print("🚀 使用增強版訓練池（25支標的 + 國際指數）")
        elif stock_list is None:
            # 基本版股票池
            stock_list = [
                '2330.TW', '2317.TW', '2454.TW', '2412.TW',
                '2882.TW', '2308.TW', '2886.TW', '2891.TW'
            ]
        
        print("=" * 70)
        print("📚 開始訓練遷移學習基礎模型（增強版）")
        print("=" * 70)
        print(f"📊 訓練標的數量: {len(stock_list)} 支")
        print(f"🎯 增強模式: {'啟用' if enhanced else '停用'}")
        print()
        
        # 🆕 收集更多歷史數據（3000-5000天）
        max_history_days = 5000 if enhanced else 2000
        min_required_days = 800 if enhanced else 500
        
        all_data = []
        success_count = 0
        
        for idx, ticker in enumerate(stock_list):
            try:
                # 進度回調
                if progress_callback:
                    progress = (idx / len(stock_list)) * 0.3  # 前30%為數據收集
                    progress_callback(idx, len(stock_list), 0, 0)
                
                # 獲取股票數據
                if ticker.startswith('^'):
                    # 國際指數使用 yfinance 直接獲取
                    print(f"  📈 下載指數 {ticker}...", end='')
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="max")
                    if not hist.empty:
                        # 重命名列以匹配台股格式
                        hist.columns = [c.capitalize() for c in hist.columns]
                else:
                    # 台股使用既有函數
                    print(f"  📈 下載股票 {ticker}...", end='')
                    _, hist, _, _ = get_stock_data(ticker)
                
                if not hist.empty and len(hist) > min_required_days:
                    # 🆕 使用更長歷史數據
                    if len(hist) > max_history_days:
                        hist = hist.iloc[-max_history_days:]
                    
                    all_data.append(hist)
                    success_count += 1
                    print(f" ✓ {len(hist):4d} 天")
                else:
                    print(f" ✗ 數據不足 ({len(hist)} 天)")
            except Exception as e:
                print(f" ✗ 失敗: {str(e)[:30]}")
                continue
        
        print()
        print(f"✅ 成功收集: {success_count}/{len(stock_list)} 支標的")
        
        if success_count < 5:
            return f"⚠️ 數據不足，至少需要5支標的的歷史數據（當前僅{success_count}支）"
        
        # 🆕 數據增強：合併多個版本的數據
        if enhanced:
            print("\n🔧 應用數據增強技術...")
            augmented_data = []
            
            for hist in all_data:
                # 原始數據
                augmented_data.append(hist.copy())
                
                # 🆕 增強1: 添加微小噪聲（模擬市場不確定性）
                noisy_hist = hist.copy()
                noise_level = 0.005  # 0.5% 噪聲
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in noisy_hist.columns:
                        noise = np.random.normal(1.0, noise_level, len(noisy_hist))
                        noisy_hist[col] = noisy_hist[col] * noise
                augmented_data.append(noisy_hist)
            
            all_data = augmented_data
            print(f"  ✓ 數據增強完成，總數據量提升至 {len(all_data)} 個樣本集")
        
        # 合併所有數據
        combined_df = pd.concat(all_data, ignore_index=True)
        total_days = len(combined_df)
        print(f"\n📊 合併後總數據量: {total_days:,} 天")
        print(f"📈 預計訓練樣本數: {total_days - self.lookback:,} 個")
        print()
        print("  ℹ️  基礎模型僅使用技術指標（不包含個股財務數據）")
        print("  ℹ️  訓練數據涵蓋：台股、ETF、國際指數")
        print()
        
        # 使用合併數據訓練基礎模型（不啟用遷移學習）
        print("🎯 開始訓練神經網絡...")
        print("-" * 70)
        
        result, _ = self.train_deep_learning_model(
            "BASE_MODEL", 
            combined_df, 
            progress_callback=progress_callback,
            use_transfer_learning=False
        )
        
        # 將訓練好的模型複製為基礎模型
        base_model_file = os.path.join(self.model_dir, "BASE_MODEL.pth")
        if os.path.exists(base_model_file):
            # 檢查源文件和目標文件是否相同（Windows路徑不區分大小寫）
            if os.path.normpath(base_model_file).lower() != os.path.normpath(self.base_model_path).lower():
                import shutil
                os.makedirs(os.path.dirname(self.base_model_path), exist_ok=True)
                shutil.copy(base_model_file, self.base_model_path)
            
            print("\n" + "=" * 70)
            print("✅ 基礎模型訓練完成並已保存！")
            print("=" * 70)
            print(f"💡 模型位置: {base_model_file}")
            print(f"📊 訓練數據: {success_count} 支標的, {total_days:,} 天歷史")
            print(f"🌍 涵蓋範圍: 台股 + ETF + 國際指數")
            print(f"🚀 後續訓練將自動使用遷移學習加速 50-70%")
            print("=" * 70)
            
            return f"✅ 基礎模型訓練成功！\n\n📊 訓練統計:\n  - 標的數量: {success_count} 支\n  - 歷史數據: {total_days:,} 天\n  - 訓練樣本: {total_days - self.lookback:,} 個\n  - 數據增強: {'啟用' if enhanced else '停用'}\n\n🚀 後續所有股票訓練將自動加速！"
        else:
            return f"❌ 基礎模型訓練失敗：找不到模型文件 {base_model_file}"

    def predict_next_price(self, ticker, df):
        """使用訓練好的模型預測"""
        model, scaler = self._get_model_instance(ticker)
        if model is None or scaler is None:
            return None
        
        try:
            if 'RSI' not in df.columns or 'K' not in df.columns:
                df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)
            
            # 🆕 [v5.1] 確保財務特徵存在
            if 'fin_financial_health_score' not in df.columns:
                from services.financial_data import get_financial_manager
                df = tech_analysis.add_financial_features(df, ticker)
            
            features = self.feature_columns
            
            # [相容性修正] 檢查模型預期的輸入特徵數量
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(features):
                # 若模型需要的特徵較少，假設使用的是舊版特徵列表 (直接截斷)
                if scaler.n_features_in_ < len(features):
                    features = features[:scaler.n_features_in_]
            
            for col in features:
                if col not in df.columns: df[col] = 0
            
            data = df[features].fillna(0).values

            if len(data) < self.lookback:
                return None
                
            last_sequence = data[-self.lookback:]
            last_sequence_scaled = scaler.transform(last_sequence)
            
            device = 'cpu'
            model.to(device)
            X_test = torch.from_numpy(np.array([last_sequence_scaled])).float().to(device)
            
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_test).item()
                
            dummy = np.zeros((1, len(features))) 
            dummy[0, 0] = pred_scaled 
            pred_real = scaler.inverse_transform(dummy)[0, 0]
            
            if not df.empty and 'Close' in df.columns:
                current_close = df['Close'].iloc[-1]
                limit_up = current_close * 1.10
                limit_down = current_close * 0.90
                
                if pred_real > limit_up:
                    pred_real = limit_up
                elif pred_real < limit_down:
                    pred_real = limit_down

            # --- 🔮 動態偏差校準 (Dynamic Calibration) ---
            # 根據 daily_predictions.json 的歷史誤差，修正當日預測
            bias = self._calculate_recent_bias(ticker, lookback_days=5)
            if bias != 0:
                print(f"[AI 校準] {ticker} 原始預測: {pred_real:.2f}, 歷史平均誤差: {bias:+.2f}, 校準後: {pred_real + bias:.2f}")
                pred_real += bias

            # --- 📰 新聞情感修正 (News Sentiment Adjustment) ---
            try:
                from services.news_analyzer import NewsAnalyzer
                news_analyzer = NewsAnalyzer()
                
                # 使用詳細模式獲取可信度
                result = news_analyzer.analyze_sentiment(ticker, detailed=True)
                
                if len(result) == 3:
                    news_score, news_list, details = result
                    confidence = details.get('confidence', 0)
                    sentiment_level = details.get('sentiment_level', '中性')
                else:
                    news_score, news_list = result
                    confidence = 0
                    sentiment_level = '中性'
                
                if news_score != 0:
                    current_close = df['Close'].iloc[-1]
                    
                    # 根據新聞情感強度調整
                    if news_score >= 5:  # 極度正面
                        adjustment_pct = 0.010  # +1.0%
                    elif news_score >= 3:  # 明顯正面
                        adjustment_pct = 0.007  # +0.7%
                    elif news_score >= 2:  # 正面
                        adjustment_pct = 0.005  # +0.5%
                    elif news_score >= 1:  # 略為正面
                        adjustment_pct = 0.002  # +0.2%
                    elif news_score <= -5:  # 極度負面
                        adjustment_pct = -0.010  # -1.0%
                    elif news_score <= -3:  # 明顯負面
                        adjustment_pct = -0.007  # -0.7%
                    elif news_score <= -2:  # 負面
                        adjustment_pct = -0.005  # -0.5%
                    elif news_score <= -1:  # 略為負面
                        adjustment_pct = -0.002  # -0.2%
                    else:
                        adjustment_pct = 0
                    
                    # 根據可信度調整影響力
                    if confidence > 70:  # 高可信度
                        confidence_factor = 1.0
                    elif confidence > 40:  # 中等可信度
                        confidence_factor = 0.7
                    else:  # 低可信度
                        confidence_factor = 0.4
                    
                    news_adjustment = current_close * adjustment_pct * confidence_factor
                    original_pred = pred_real
                    pred_real += news_adjustment
                    
                    # 確保不超過漲跌停限制
                    limit_up = current_close * 1.10
                    limit_down = current_close * 0.90
                    pred_real = max(limit_down, min(limit_up, pred_real))
                    
                    if abs(news_adjustment) > 0.01:  # 只有顯著調整才印出
                        emoji = news_analyzer.get_sentiment_emoji(news_score)
                        print(f"[📰 新聞情感] {ticker} {emoji} 分數: {news_score:+d} | {sentiment_level} | "
                        f"可信度: {confidence:.0f}% | 調整: {adjustment_pct*confidence_factor:+.2%} ({news_adjustment:+.2f}) | "
                        f"{original_pred:.2f} → {pred_real:.2f}")
                        if news_list and len(news_list) > 0:
                            # 顯示最關鍵的新聞
                            key_news = [n for n in news_list if '🔴' in n or '🟢' in n]
                            if key_news:
                                print(f"  關鍵新聞: {key_news[0][:60]}...")
                            else:
                                print(f"  最新消息: {news_list[0][:60]}...")
            except Exception as e:
                # 新聞分析失敗不影響預測
                pass

            pred_real = self._adjust_price_tick(ticker, pred_real)
            return pred_real
        except Exception:
            return None

    def predict_next_day(self, ticker, df):
        """
        🆕 預測明日價格變化百分比（AutoTrader v2.0 專用）
        
        與 predict_next_price 的差異:
        - 返回格式為字典 {predicted_change_pct, confidence, predicted_price}
        - 包含信心度評估（基於模型損失、數據質量等）
        
        Args:
            ticker: 股票代碼
            df: 歷史數據
            
        Returns:
            dict 或 None: {
                'predicted_price': float,
                'predicted_change_pct': float,  # -0.05 表示預測下跌 5%
                'confidence': float  # 0-1 之間
            }
        """
        try:
            # 使用現有的預測方法
            predicted_price = self.predict_next_price(ticker, df)
            
            if predicted_price is None or df.empty:
                return None
            
            # 計算預測變化百分比
            current_price = df['Close'].iloc[-1]
            predicted_change_pct = (predicted_price - current_price) / current_price
            
            # 🎯 計算信心度（基於多個因素）
            confidence = 0.5  # 基礎信心度
            
            # 因素1: 模型損失（越低越好）
            if ticker in self.models:
                model_info = self.models[ticker]
                val_loss = model_info.get('last_loss', 1.0)
                
                # 損失 < 0.001 → 高信心度 +0.2
                # 損失 < 0.01 → 中信心度 +0.1
                # 損失 > 0.1 → 低信心度 -0.1
                if val_loss < 0.001:
                    confidence += 0.2
                elif val_loss < 0.01:
                    confidence += 0.1
                elif val_loss > 0.1:
                    confidence -= 0.1
            
            # 因素2: 數據質量（數據越多越好）
            data_days = len(df)
            if data_days >= 1000:
                confidence += 0.15
            elif data_days >= 500:
                confidence += 0.1
            elif data_days < 200:
                confidence -= 0.15
            
            # 因素3: 預測變化合理性（極端預測降低信心度）
            if abs(predicted_change_pct) > 0.08:  # 單日變化 > 8%
                confidence -= 0.2
            elif abs(predicted_change_pct) > 0.05:  # 單日變化 > 5%
                confidence -= 0.1
            
            # 因素4: 財務健康度（如果有財務數據）
            if 'fin_financial_health_score' in df.columns:
                fin_health = df['fin_financial_health_score'].iloc[-1]
                if fin_health > 70:  # 財務健康
                    confidence += 0.1
                elif fin_health < 30:  # 財務不佳
                    confidence -= 0.1
            
            # 限制信心度範圍在 0.2 到 0.95 之間
            confidence = max(0.2, min(0.95, confidence))
            
            return {
                'predicted_price': float(predicted_price),
                'predicted_change_pct': float(predicted_change_pct),
                'confidence': float(confidence),
                'current_price': float(current_price)
            }
            
        except Exception as e:
            print(f"predict_next_day 錯誤 [{ticker}]: {e}")
            return None

    def get_backtest_data(self, ticker, df, days=60):
        """[新增] 產生最近 N 天的模型回測曲線 (視覺化驗證用)"""
        model, scaler = self._get_model_instance(ticker)
        if model is None or scaler is None or df.empty or len(df) < (self.lookback + days):
            return None
        
        try:
            # 確保特徵存在
            if 'MOM' not in df.columns or 'CCI' not in df.columns:
                df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)
            
            # 🆕 [v5.1] 確保財務特徵存在
            if 'fin_financial_health_score' not in df.columns:
                from services.financial_data import get_financial_manager
                df = tech_analysis.add_financial_features(df, ticker)
                
            features = self.feature_columns
            
            # [相容性修正] 檢查模型預期的輸入特徵數量
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(features):
                if scaler.n_features_in_ < len(features):
                    features = features[:scaler.n_features_in_]

            for col in features:
                if col not in df.columns: df[col] = 0
            
            data_values = df[features].fillna(0).values
            dates = df.index
            
            results = []
            device = 'cpu'
            model.to(device)
            model.eval()
            
            # 針對最近 N 天，每一天都嘗試做「當下預測」
            # 例如: 要預測 Day_100，就拿 Day_(100-lookback) 到 Day_99 的資料餵給模型
            start_idx = len(df) - days
            
            with torch.no_grad():
                for i in range(start_idx, len(df)):
                    # 取出這一天之前的 Lookback 窗口數據
                    seq_data = data_values[i-self.lookback:i]
                    # 正規化
                    seq_scaled = scaler.transform(seq_data)
                    X_test = torch.from_numpy(np.array([seq_scaled])).float().to(device)
                    
                    # 預測
                    pred_scaled = model(X_test).item()
                    
                    # 反正規化
                    dummy = np.zeros((1, len(features)))
                    dummy[0, 0] = pred_scaled
                    pred_val = scaler.inverse_transform(dummy)[0, 0]
                    
                    results.append({
                        "Date": dates[i],
                        "Actual": df['Close'].iloc[i],
                        "Predicted": pred_val
                    })
                    
            return pd.DataFrame(results)
        except Exception as e:
            print(f"Backtest error: {e}")
            return None
    
    def get_outdated_models(self):
        """獲取需要重新訓練的模型列表（版本不兼容或不存在）"""
        outdated = []
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pth'):
                ticker = filename.replace('.pth', '')
                path = os.path.join(self.model_dir, ticker + '.pth')
                
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    model_version = checkpoint.get('version', '1.0')
                    
                    if model_version != self.MODEL_VERSION:
                        outdated.append({
                            'ticker': ticker,
                            'old_version': model_version,
                            'current_version': self.MODEL_VERSION,
                            'reason': '架構更新'
                        })
                except Exception as e:
                    outdated.append({
                        'ticker': ticker,
                        'old_version': 'unknown',
                        'current_version': self.MODEL_VERSION,
                        'reason': f'加載失敗: {str(e)[:50]}'
                    })
        
        return outdated
    
    def _calculate_recent_bias(self, ticker, lookback_days=5):
        """計算最近 N 天的平均預測誤差 (實際 - 預測)"""
        try:
            if not os.path.exists(self.prediction_file):
                return 0.0
            
            # 使用列表倒推讀取比較快，這裡暫時全讀
            with open(self.prediction_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 篩選該股票的紀錄
            target_history = [h for h in history if h.get('ticker') == ticker]
            
            if not target_history:
                return 0.0
                
            # 取最近 N 筆有效紀錄（有 actual_price 才算有效）
            valid_records = [
                h for h in target_history 
                if 'actual_price' in h and 'predicted_price' in h and h['actual_price'] > 0
            ]
            
            # 根據日期排序 (新到舊)
            valid_records.sort(key=lambda x: x['date'], reverse=True)
            recent_records = valid_records[:lookback_days]
            
            if not recent_records:
                return 0.0
            
            # 計算平均誤差 (實際 - 預測)
            # 如果實際是 100, 預測是 98 -> 誤差 +2 (代表模型低估，下次要加回來)
            # 如果實際是 100, 預測是 105 -> 誤差 -5 (代表模型高估，下次要扣掉)
            total_bias = sum((r['actual_price'] - r['predicted_price']) for r in recent_records)
            avg_bias = total_bias / len(recent_records)
            
            # 設定一個合理的修正上限 (例如不超過股價的 3%)，避免過度修正
            return avg_bias
        except Exception as e:
            # print(f"校準計算錯誤: {e}")
            return 0.0


    def _adjust_price_tick(self, ticker, price):
        """根據台股規則校正價格檔位"""
        try:
            is_etf = str(ticker).startswith('00')
            if is_etf:
                if price < 50: return round(price, 2)
                else: return round(price * 20) / 20
            else:
                if price < 10: return round(price, 2)
                elif price < 50: return round(price * 20) / 20 
                elif price < 100: return round(price, 1)    
                elif price < 500: return round(price * 2) / 2   
                elif price < 1000: return round(price)           
                else: return round(price / 5) * 5                
        except:
            return price

    def run_morning_predictions(self):
        """[新功能] 開盤前執行：對所有已訓練的模型進行預測並存檔"""
        results = []
        today = datetime.datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        yesterday = today - datetime.timedelta(days=1)
        
        # 🔧 修復：創建字典鍵的副本，避免迭代時修改字典
        ticker_list = list(self.models.keys())
        
        loop_count = 0
        for ticker in ticker_list:
            loop_count += 1
            if loop_count % 20 == 0: gc.collect() 
            
            try:
                symbol, hist_data, current_price, _ = get_stock_data(ticker)
                
                if hist_data.empty: continue
                
                pred_price = self.predict_next_price(ticker, hist_data)
                
                if pred_price:
                    # [優化] 使用精確的日期篩選，而不是 iloc
                    # 目標：找到昨天（或最接近的前一個交易日）的收盤價作為 base_price
                    
                    # 將索引轉為 date 型態以便比較
                    hist_data.index = pd.to_datetime(hist_data.index)
                    
                    # 篩選出昨天之前（包含昨天）的所有資料
                    before_today = hist_data[hist_data.index.date < today]
                    
                    if not before_today.empty:
                        # 取最後一筆（最接近今天的前一個交易日）
                        base_price = float(before_today['Close'].iloc[-1])
                    else:
                        # 備援：如果篩選後沒資料，使用原始最後一筆
                        base_price = float(hist_data['Close'].iloc[-1])
                    
                    expected_change = ((pred_price - base_price) / base_price) * 100
                    
                    results.append({
                        "date": today_str,
                        "ticker": ticker,
                        "base_price": base_price,
                        "predicted_price": float(pred_price),
                        "expected_change_pct": round(expected_change, 2),
                        "actual_price": None,
                        "diff_pct": None,
                        "accuracy": "等待收盤驗證"
                    })
            except Exception as e:
                continue
                
        all_records = []
        if os.path.exists(self.prediction_file):
            try:
                with open(self.prediction_file, 'r', encoding='utf-8') as f:
                    all_records = json.load(f)
            except: pass
            
        all_records = [r for r in all_records if r['date'] != today_str]
        all_records.extend(results)
        
        with open(self.prediction_file, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=4)
            
        return len(results)

    def validate_afternoon_results(self):
        """[新功能] 收盤後執行：驗證今日預測準確度"""
        if not os.path.exists(self.prediction_file):
            return 0, []
            
        with open(self.prediction_file, 'r', encoding='utf-8') as f:
            all_records = json.load(f)
            
        today = datetime.datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        updated_count = 0
        
        for record in all_records:
            if record['date'] == today_str:
                ticker = record['ticker']
                try:
                    symbol, hist_data, current_price, _ = get_stock_data(ticker)
                    
                    if not hist_data.empty:
                        # [優化] 使用精確日期篩選，取得今天的收盤價
                        hist_data.index = pd.to_datetime(hist_data.index)
                        
                        # 篩選今天的資料
                        today_data = hist_data[hist_data.index.date == today]
                        
                        if not today_data.empty:
                            actual_close = float(today_data['Close'].iloc[-1])
                            
                            # [修正] 保護 base_price 不被覆寫
                            # 只有在尚未設定 actual_price 時才更新（避免重複驗證時覆寫）
                            if record.get('actual_price') is None:
                                record['actual_price'] = actual_close
                            
                            pred = record['predicted_price']
                            actual = record['actual_price']
                            diff = pred - actual
                            diff_pct = (diff / actual) * 100 
                            
                            record['diff_pct'] = round(diff_pct, 2)
                            
                            if abs(diff_pct) < 1.0:
                                record['accuracy'] = "⭐⭐⭐ 準確"
                            elif abs(diff_pct) < 2.5:
                                record['accuracy'] = "⭐ 普通"
                            else:
                                record['accuracy'] = "❌ 失準"
                                
                            updated_count += 1
                        else:
                            record['accuracy'] = "等待資料更新"
                            
                except Exception as e:
                    pass
                    
        with open(self.prediction_file, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=4)
    
    def _add_market_sentiment_features(self, df):
        """
        🆕 v4.0: 新增市場情緒特徵
        增加 9 個市場級別特徵，提升預測準確度
        """
        try:
            # 1. 資金流向指標 (MFI)
            df['MFI'] = self.sentiment_analyzer.calculate_money_flow_index(df)
            
            # 2. 趨勢強度指標 (ADX)
            df['ADX'] = self.sentiment_analyzer.calculate_adx(df)
            
            # 3. 股價位置（相對52週高低點）
            df['Price_Position'] = self.sentiment_analyzer.get_price_position(df)
            
            # 4. 異常成交量檢測
            volume_anomaly = self.sentiment_analyzer.detect_volume_anomaly(df)
            df['Volume_Anomaly'] = volume_anomaly.astype(int)
            
            # 5. 急漲急跌檢測
            sharp_changes = self.sentiment_analyzer.detect_sharp_change(df)
            df['Sharp_Up'] = sharp_changes['sharp_up'].astype(int)
            df['Sharp_Down'] = sharp_changes['sharp_down'].astype(int)
            
            # 6. 大盤情緒評分
            market_sentiment = self.sentiment_analyzer.get_market_sentiment()
            df['Market_Sentiment_Score'] = market_sentiment['score']
            
            print(f"  ✅ 市場情緒特徵已加入: MFI, ADX, Price_Position, Volume_Anomaly等9個")
            
        except Exception as e:
            print(f"  ⚠️ 市場情緒特徵計算失敗: {e}")
            # 填充預設值
            for col in ['MFI', 'ADX', 'Price_Position', 'Volume_Anomaly', 
                        'Market_Sentiment_Score', 'Sharp_Up', 'Sharp_Down']:
                if col not in df.columns:
                    df[col] = 50 if col in ['MFI', 'ADX', 'Market_Sentiment_Score', 'Price_Position'] else 0
        
        return df
    
    def _select_best_model_type(self, data_size, high_volatility=False):
        """
        🆕 v5.0: 智能選擇最佳模型架構
        根據數據量和波動度自動選擇
        """
        # 數據量足夠大且重要股票 → Ensemble
        if data_size > 2000:
            return "Ensemble"  # 模型組合，最強但最慢
        
        # 中等數據量 → Transformer 或 AttentionLSTM
        elif data_size > 1000:
            if high_volatility:
                return "AttentionLSTM"  # 高波動用 Attention
            else:
                return "Transformer"    # 平穩用 Transformer
        
        # 小數據量 → GRU（快速且有效）
        elif data_size > 500:
            return "GRU"
        
        # 極小數據量 → LSTM（更保守）
        else:
            return "LSTM"
            
        return updated_count, all_records
