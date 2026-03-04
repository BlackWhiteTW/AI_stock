"""
🕸️ GNN 市場關係訓練服務 (ST-GNN Edition) v2.0
整合時空特徵與歷史滑動窗口訓練
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from services.ai_model_defs import SpatioTemporalGNN
from services import tech_analysis
from utils import get_stock_data, get_tw_stock_list

class GNNService:
    def __init__(self):
        self.model_dir = "data/saved_models/gnn"
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join(self.model_dir, "market_stgnn.pth")
        
        self.lookback = 20 # 序列長度
        self.feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'Volatility', 
            'Market_Sentiment_Score', 'Price_Velocity', 'Relative_Strength'
        ]

    def prepare_historical_tensor(self, tickers, days=250):
        """
        🚀 核心：構建歷史三維張量 [Time, Nodes, Features]
        """
        print(f"📊 正在準備歷史時空數據 (範圍: {days} 天)...")
        
        # 1. 對齊所有股票的歷史數據
        stock_dfs = {}
        all_dates = None
        
        for ticker in tickers:
            try:
                _, df, _, _ = get_stock_data(ticker)
                if len(df) < days + self.lookback: continue
                
                # 計算所有強化特徵
                df = tech_analysis.calculate_technical_indicators(df, ticker=ticker)
                df = df.tail(days + self.lookback)
                
                stock_dfs[ticker] = df
                if all_dates is None:
                    all_dates = df.index
                else:
                    all_dates = all_dates.intersection(df.index)
            except: continue
            
        valid_tickers = list(stock_dfs.keys())
        all_dates = sorted(list(all_dates))
        
        # 2. 填充三維張量 [Time, Nodes, Features]
        num_time = len(all_dates)
        num_nodes = len(valid_tickers)
        num_feats = len(self.feature_columns)
        
        data_tensor = np.zeros((num_time, num_nodes, num_feats))
        labels_tensor = np.zeros((num_time, num_nodes))
        
        for i, date in enumerate(all_dates):
            for j, ticker in enumerate(valid_tickers):
                df = stock_dfs[ticker]
                # 取得當天特徵
                row = df.loc[date]
                data_tensor[i, j, :] = row[self.feature_columns].values
                
                # 取得 T+1 漲跌幅作為 Label
                try:
                    next_idx = df.index.get_loc(date) + 1
                    if next_idx < len(df):
                        future_ret = (df['Close'].iloc[next_idx] - row['Close']) / row['Close']
                        labels_tensor[i, j] = future_ret
                except: pass
                
        return data_tensor, labels_tensor, valid_tickers

    def train_market_gnn(self, tickers=None, epochs=50):
        """
        使用滑動窗口訓練 ST-GNN
        """
        if tickers is None:
            df = get_tw_stock_list()
            tickers = df['Ticker'].head(100).tolist() # 先以 100 支為試點
            
        # 1. 準備數據
        raw_data, raw_labels, valid_tickers = self.prepare_historical_tensor(tickers)
        adj = self.build_market_graph(valid_tickers)
        
        # 2. 構建滑動窗口樣本
        # X: [Samples, Seq_Len, Nodes, Features]
        # Y: [Samples, Nodes, 1]
        X, Y = [], []
        for i in range(len(raw_data) - self.lookback - 1):
            X.append(raw_data[i : i + self.lookback])
            Y.append(raw_labels[i + self.lookback])
            
        X = torch.from_numpy(np.array(X)).float().to(self.device)
        Y = torch.from_numpy(np.array(Y)).float().to(self.device).unsqueeze(-1)
        
        print(f"🔥 訓練樣本構建完成: {X.shape}")
        
        # 3. 訓練
        model = SpatioTemporalGNN(
            node_features=len(self.feature_columns),
            hidden_dim=128,
            seq_len=self.lookback,
            output_dim=1
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        
        model.train()
        for epoch in range(epochs):
            # 這裡我們使用全量訓練 (如果 VRAM 不夠需改用 DataLoader)
            optimizer.zero_grad()
            output = model(X, adj) # [Samples, Nodes, 1]
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
                
        # 4. 保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'tickers': valid_tickers,
            'features': self.feature_columns,
            'lookback': self.lookback
        }, self.model_path)
        
        return f"✅ ST-GNN 歷史訓練完成！涵蓋 {len(valid_tickers)} 支股票 x {len(X)} 個時間樣本。"

    def build_market_graph(self, tickers):
        """構建市場圖結構 (基於產業)"""
        num_nodes = len(tickers)
        adj = np.eye(num_nodes)
        df_stocks = get_tw_stock_list()
        
        industry_map = {}
        for ticker in tickers:
            code = ticker.split('.')[0]
            row = df_stocks[df_stocks['代碼'] == code]
            if not row.empty:
                ind = row.iloc[0]['產業']
                if ind not in industry_map: industry_map[ind] = []
                industry_map[ind].append(ticker)
        
        t_to_idx = {t: i for i, t in enumerate(tickers)}
        for ind, members in industry_map.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if members[i] in t_to_idx and members[j] in t_to_idx:
                        idx_i, idx_j = t_to_idx[members[i]], t_to_idx[members[j]]
                        adj[idx_i, idx_j] = 1.0; adj[idx_j, idx_i] = 1.0
        
        # 正規化
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        return torch.from_numpy(np.diag(d_inv_sqrt).dot(adj).dot(np.diag(d_inv_sqrt))).float().to(self.device)

    def prepare_node_features(self, tickers):
        # 為了相容性保留此方法，但在 v2.0 中主要使用 prepare_historical_tensor
        return None, None, tickers

    def predict_market_relations(self):
        """使用訓練好的 ST-GNN 預測明日強度"""
        if not os.path.exists(self.model_path): return "❌ 未訓練"
        checkpoint = torch.load(self.model_path, map_location=self.device)
        tickers = checkpoint['tickers']
        
        # 準備最後一個窗口的數據
        raw_data, _, valid_tickers = self.prepare_historical_tensor(tickers, days=1)
        adj = self.build_market_graph(valid_tickers)
        
        # 取最後 lookback 天
        x_input = torch.from_numpy(raw_data[-self.lookback:]).float().to(self.device).unsqueeze(0)
        
        model = SpatioTemporalGNN(len(self.feature_columns), 128, self.lookback, 1).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            preds = model(x_input, adj)
            
        return dict(zip(valid_tickers, preds.cpu().numpy().flatten()))
