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

class AIService:
    """
    🚀 高性能 AI 預測服務 v5.3 (2026 旗艦版)
    專注於 Residual Attention LSTM 與 Alpha 特徵融合
    """
    MODEL_VERSION = "5.3-LSTM-Attention"
    
    def __init__(self):
        self.models = {} # Metadata
        self.loaded_models = {} # LRU Cache
        self.MAX_CACHE_SIZE = 15
        
        self.sentiment_analyzer = MarketSentimentAnalyzer() 
        self.lookback = 60 
        self.model_dir = "data/saved_models"
        self.prediction_file = "data/daily_predictions.json"
        self.base_model_path = "data/saved_models/base_model.pth"
        
        # 🆕 v5.3 完整 26 個核心與 Alpha 特徵
        self.feature_columns = [
            # 基礎價量 (5)
            'Close', 'Open', 'High', 'Low', 'Volume',
            # 技術指標 (6)
            'RSI', 'MACD', 'K', 'D', 'ATR', 'Volatility',
            # 市場情緒 (5)
            'MFI', 'ADX', 'Price_Position', 'Volume_Anomaly', 'Market_Sentiment_Score',
            # Alpha 動能 (4)
            'Price_Velocity', 'Price_Acceleration', 'OBV', 'BIAS_20',
            # 財務與相對強度 (6)
            'Relative_Strength', 'Dist_MA20', 'fin_roe_normalized', 
            'fin_debt_ratio_normalized', 'fin_pe_ratio_normalized', 'fin_dividend_yield_normalized'
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
            for etf in ['0050.TW', '0051.TW', '0056.TW']:
                holding_data = scraper.get_holding(etf)
                if holding_data and 'top_holdings' in holding_data:
                    for item in holding_data['top_holdings']:
                        raw_id = item.get('代號') or item.get('股票代號') or item.get('股票', '')
                        clean_id = ''.join(filter(str.isdigit, str(raw_id)))
                        if len(clean_id) == 4: # 只選一般個股，排除權證
                            ticker = f"{clean_id}.TW"
                            if ticker not in stock_list: stock_list.append(ticker)
            
            # 2. 產業均衡補充 (從本地 CSV 各產業分散選取)
            if len(stock_list) < count and os.path.exists('data/tw_stock_list.csv'):
                try:
                    df_all = pd.read_csv('data/tw_stock_list.csv')
                    # 清洗數據：只留 4 位數代號
                    df_all['Ticker_Len'] = df_all['代碼'].astype(str).str.len()
                    df_clean = df_all[df_all['Ticker_Len'] == 4].copy()
                    
                    # 按產業分組，每組取前 N 支 (通常 CSV 已按規模排序)
                    industries = df_clean['產業'].unique()
                    per_industry = max(1, (count - len(stock_list)) // len(industries) + 1)
                    
                    for ind in industries:
                        ind_stocks = df_clean[df_clean['產業'] == ind].head(per_industry)
                        for _, row in ind_stocks.iterrows():
                            ticker = str(row['Ticker'])
                            if ticker not in stock_list:
                                stock_list.append(ticker)
                            if len(stock_list) >= count: break
                        if len(stock_list) >= count: break
                except Exception as e:
                    print(f"⚠️ 產業抽樣失敗: {e}")
        
        except Exception as e:
            print(f"⚠️ 獲取代表性標的時發生錯誤: {e}")
        
        # 3. 確保至少有保底清單
        base_stocks = ['2330.TW', '2317.TW', '2454.TW', '2303.TW', '2308.TW', '2603.TW', '2881.TW', '2882.TW']
        for s in base_stocks:
            if s not in stock_list: stock_list.insert(0, s)
        
        final_list = stock_list[:count]
        print(f"✅ 已選定 {len(final_list)} 支各產業代表性標的")
        return final_list

    def _load_all_models(self):
        """掃描本地磁碟中的模型檔案"""
        self.models = {}
        if not os.path.exists(self.model_dir): return
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pth"):
                ticker = filename.replace(".pth", "")
                self.models[ticker] = {"last_loss": 0.0}

    def _get_model_instance(self, ticker):
        """[LRU Cache] 從記憶體或硬碟加載模型"""
        if ticker in self.loaded_models:
            val = self.loaded_models.pop(ticker)
            self.loaded_models[ticker] = val
            return val['model'], val['scaler']
        
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        if not os.path.exists(path): return None, None
            
        try:
            # 兼容性加載
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            input_dim = checkpoint.get('input_dim', len(self.feature_columns))
            hidden_dim = checkpoint.get('hidden_dim', 256)
            num_layers = checkpoint.get('num_layers', 3)
            
            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim=1)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            if len(self.loaded_models) >= self.MAX_CACHE_SIZE:
                del self.loaded_models[next(iter(self.loaded_models))]
            
            self.loaded_models[ticker] = {"model": model, "scaler": checkpoint['scaler']}
            return model, checkpoint['scaler']
        except Exception as e:
            print(f"⚠️ 模型 {ticker} 加載失敗: {e}")
            return None, None

    def prepare_lstm_data(self, df, lookback, ticker=None):
        """準備與標準化 LSTM 數據"""
        df = self._add_market_sentiment_features(df)
        
        # 確保特徵完整計算
        if 'Price_Velocity' not in df.columns:
            df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)
        
        # 處理缺失特徵
        for col in self.feature_columns:
            if col not in df.columns: df[col] = 0
                
        data_clean = df[self.feature_columns].fillna(0).iloc[60:].copy()
        if data_clean.empty: data_clean = df[self.feature_columns].fillna(0).copy()
        
        # 高波動股票對數處理
        price_cv = data_clean['Close'].std() / data_clean['Close'].mean()
        if price_cv > 0.6:
            for feat in ['Close', 'Open', 'High', 'Low']:
                data_clean[feat] = np.log1p(data_clean[feat])
            
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data_clean.values)
        
        X, y = [], []
        for i in range(len(scaled_data) - lookback - 1):
            X.append(scaled_data[i : i + lookback])
            y.append(scaled_data[i + lookback, 0]) # 預測目標：Close
            
        return np.array(X), np.array(y), scaler

    def train_deep_learning_model(self, ticker, df, progress_callback=None, use_transfer_learning=True):
        """
        🔥 專業級 LSTM 訓練引擎
        包含 GPU 加速、SWA、方向性損失優化
        """
        if len(df) > 3000: df = df.iloc[-3000:].copy()
        if len(df) < 200: return "⚠️ 數據不足 (至少需 200 天)", 0

        # 特徵工程與回測優化
        print(f"  📊 正在為 {ticker} 進行特徵工程...")
        df = tech_analysis.calculate_technical_indicators(df, ticker=ticker, include_financial=True)
        self.strategy.optimize_strategy_basic(ticker, df)
        
        X, y, scaler = self.prepare_lstm_data(df, self.lookback, ticker=ticker)
        
        # 數據分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 硬體偵測與優化配置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 128 if torch.cuda.is_available() else 32
        num_workers = 2 if torch.cuda.is_available() else 0
        
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
            batch_size=batch_size * 2, shuffle=False
        )

        # 模型初始化
        input_dim = X.shape[2]
        hidden_dim = 256 if len(X) > 1000 else 128
        num_layers = 3 if len(X) > 1000 else 2
        model = LSTMModel(input_dim, hidden_dim, num_layers, 1).to(device)
        
        # 🚀 遷移學習核心
        if use_transfer_learning and os.path.exists(self.base_model_path):
            try:
                base = torch.load(self.base_model_path, map_location=device, weights_only=False)
                if base['input_dim'] == input_dim:
                    model.load_state_dict(base['model_state_dict'], strict=False)
                    print(f"  ✓ 已從主模型繼承權重")
            except: pass

        # 損失函數與優化器 (Huber + Directional Penalty)
        criterion_base = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        # SWA 設置
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.0001)
        swa_start = 70

        best_val_loss = float('inf')
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
                directional_loss = torch.mean(direction_mismatch.float() * (outputs - targets)**2) * 0.5
                
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
                    progress_callback(epoch + 1, epochs, avg_val_loss, time.time() - start_time)

        # 結束訓練並切換回 CPU
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        final_model = swa_model.module.to('cpu')
        self.loaded_models[ticker] = {"model": final_model, "scaler": scaler}
        
        return f"✅ {ticker} 訓練完成！Best Loss: {best_val_loss:.6f}", best_val_loss

    def train_base_model(self, stock_list=None, progress_callback=None, enhanced=True):
        """🚀 終極優化版主模型訓練 (2026 專業版)"""
        if stock_list is None:
            # 🌐 動態從網路獲取代表性標的 (增加至 200 支)
            stock_list = self._fetch_representative_stocks(count=200)
            
        print("⏬ 正在獲取大盤數據作為 Alpha 特徵參照...")
        market_index = yf.download("^TWII", period="max", progress=False)
        
        all_data = []
        total_stocks = len(stock_list)
        for idx, ticker in enumerate(stock_list):
            try:
                # 🆕 增加進度回報：通知 UI 正在下載哪支股票代碼
                if progress_callback:
                    progress_callback(idx + 1, total_stocks, 0, 0, ticker=ticker)
                
                _, hist, _, _ = get_stock_data(ticker)
                if len(hist) < 500: continue
                hist_merged = hist.join(market_index['Close'], rsuffix='_Market', how='inner')
                hist_merged.rename(columns={'Close_Market': 'Market_Close'}, inplace=True)
                all_data.append(tech_analysis.calculate_technical_indicators(hist_merged, ticker=ticker))
                print(f"✅ 已整合 {ticker}")
            except: continue
            
        if not all_data: return "❌ 數據收集失敗"
        combined_df = pd.concat(all_data, ignore_index=True)
        X, y, scaler = self.prepare_lstm_data(combined_df, self.lookback)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_dim=X.shape[2], hidden_dim=256, num_layers=3, output_dim=1).to(device)
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()), batch_size=256, shuffle=True)
        
        epochs = 150
        for epoch in range(epochs):
            model.train()
            l_sum = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); out = model(bx); loss = criterion(out, by.unsqueeze(1))
                loss.backward(); optimizer.step(); l_sum += loss.item()
            
            if progress_callback:
                progress_callback(epoch+1, epochs, l_sum/len(loader), 0)
            if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1} Loss: {l_sum/len(loader):.6f}")
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_dim': X.shape[2],
            'hidden_dim': 256,
            'num_layers': 3,
            'version': self.MODEL_VERSION
        }, self.base_model_path)
        return "✅ 主模型訓練成功，已儲存權重。"

    def predict_next_price(self, ticker, df):
        """執行價格預測"""
        model, scaler = self._get_model_instance(ticker)
        if not model or df.empty: return None
        
        try:
            df = tech_analysis.calculate_technical_indicators(df, ticker=ticker)
            features = self.feature_columns
            for col in features:
                if col not in df.columns: df[col] = 0
            
            data = df[features].fillna(0).values[-self.lookback:]
            if len(data) < self.lookback: return None
            
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
            current_close = df['Close'].iloc[-1]
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
        if pred_price is None: return None
        
        current_price = df['Close'].iloc[-1]
        change_pct = (pred_price - current_price) / current_price
        
        # 信心度估算：基於模型訓練損失
        confidence = 0.6
        if ticker in self.models:
            loss = self.models[ticker].get('last_loss', 0.1)
            if loss < 0.001: confidence = 0.85
            elif loss < 0.01: confidence = 0.75
            
        return {
            'predicted_price': float(pred_price),
            'predicted_change_pct': float(change_pct),
            'confidence': float(confidence),
            'current_price': float(current_price)
        }

    def _add_market_sentiment_features(self, df):
        """整合市場情緒特徵"""
        try:
            df['MFI'] = self.sentiment_analyzer.calculate_money_flow_index(df)
            df['ADX'] = self.sentiment_analyzer.calculate_adx(df)
            df['Price_Position'] = self.sentiment_analyzer.get_price_position(df)
            sent = self.sentiment_analyzer.get_market_sentiment()
            df['Market_Sentiment_Score'] = sent['score']
        except: pass
        return df

    def _calculate_recent_bias(self, ticker, lookback_days=5):
        """計算近期預測偏差以進行自動校準"""
        try:
            if not os.path.exists(self.prediction_file): return 0.0
            with open(self.prediction_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            target = [h for h in history if h.get('ticker') == ticker and h.get('actual_price')]
            if not target: return 0.0
            target.sort(key=lambda x: x['date'], reverse=True)
            recent = target[:lookback_days]
            if not recent: return 0.0
            return sum((r['actual_price'] - r['predicted_price']) for r in recent) / len(recent)
        except: return 0.0

    def _adjust_price_tick(self, ticker, price):
        """根據台股規則校正價格檔位"""
        try:
            if str(ticker).startswith('00'):
                return round(price, 2) if price < 50 else round(price * 20) / 20
            if price < 10: return round(price, 2)
            elif price < 50: return round(price * 20) / 20
            elif price < 100: return round(price, 1)
            elif price < 500: return round(price * 2) / 2
            elif price < 1000: return round(price)
            return round(price / 5) * 5
        except: return price

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
                    base = float(hist['Close'].iloc[-1])
                    results.append({
                        "date": today_str, "ticker": ticker, "base_price": base,
                        "predicted_price": float(pred), "expected_change_pct": round((pred-base)/base*100, 2),
                        "actual_price": None, "accuracy": "等待收盤"
                    })
            except: continue
        
        # 讀取並合併舊紀錄
        all_recs = []
        if os.path.exists(self.prediction_file):
            try:
                with open(self.prediction_file, 'r', encoding='utf-8') as f:
                    all_recs = json.load(f)
            except: pass
        all_recs = [r for r in all_recs if r['date'] != today_str] + results
        with open(self.prediction_file, 'w', encoding='utf-8') as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=4)
        return len(results)

    def validate_afternoon_results(self):
        """收盤後驗證準確度"""
        if not os.path.exists(self.prediction_file): return 0, []
        with open(self.prediction_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        count = 0
        for r in records:
            if r['date'] == today_str and not r.get('actual_price'):
                try:
                    _, hist, _, _ = get_stock_data(r['ticker'])
                    actual = float(hist['Close'].iloc[-1])
                    r['actual_price'] = actual
                    diff = abs(r['predicted_price'] - actual) / actual * 100
                    r['accuracy'] = "⭐⭐⭐" if diff < 1.0 else ("⭐" if diff < 2.5 else "❌")
                    count += 1
                except: continue
        with open(self.prediction_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        return count, records

    def _save_model(self, ticker, model, scaler, last_loss, input_dim):
        """安全儲存模型"""
        path = os.path.join(self.model_dir, f"{ticker}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'last_loss': last_loss,
            'input_dim': input_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'version': self.MODEL_VERSION
        }, path)
