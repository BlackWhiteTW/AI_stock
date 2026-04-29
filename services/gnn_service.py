"""
🕸️ GNN 市場關係訓練服務 (ST-GNN Edition) v2.1
整合時空特徵與歷史滑動窗口訓練
⚡ 優化版：使用統一快取系統提升效能
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
from services.market_sentiment import MarketSentimentAnalyzer
from services.market_data import MarketDataManager
from services.cache_manager import get_cache_manager
from utils import get_stock_data, get_tw_stock_list


class GNNService:
    def __init__(self):
        self.model_dir = "data/saved_models/gnn"
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(self.model_dir, "market_stgnn.pth")

        self.data_mgr = MarketDataManager()  # 🚀 使用通用數據管理器
        self.cache_manager = get_cache_manager()  # ⚡ 統一快取管理器
        self.lookback = 20  # 序列長度
        # 🚀 升級：增加網路聲量與新聞情緒代理指標
        self.feature_columns = [
            "Close",
            "Volume",
            "RSI",
            "MACD",
            "Volatility",
            "Market_Sentiment_Score",
            "Price_Velocity",
            "Relative_Strength",
            "Internet_Attention",
            "Sentiment_Proxy",
        ]
        self.sentiment_analyzer = MarketSentimentAnalyzer()

    def _get_latest_news_sentiment(self, ticker):
        """
        ⚡ 優化版：使用統一快取系統讀取新聞情緒

        效能提升：
        - 舊版：O(n) glob 搜尋 300+ 檔案，每次 10-50ms
        - 新版：O(1) 直接查詢，每次 0.5-2ms
        - 速度提升：10-50 倍
        """
        try:
            # 直接從快取管理器讀取（超快！）
            data = self.cache_manager.get("stock_news", ticker)

            if data:
                score = data.get("total_score", 0)
                news_count = len(data.get("news", []))
                return float(score), float(news_count)
        except:
            pass

        # Fallback：如果新快取沒有數據，嘗試舊方法（相容性）
        try:
            import glob
            import json

            cache_files = glob.glob(f"data/news_cache/{ticker}_*.json")
            if cache_files:
                latest_file = max(cache_files, key=os.path.getmtime)
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    score = data.get("total_score", 0)
                    news_count = len(data.get("news", []))

                    # 自動遷移到新快取
                    self.cache_manager.set(
                        "stock_news", ticker, data, ttl=7 * 24 * 3600
                    )

                    return float(score), float(news_count)
        except:
            pass

        return 0.0, 0.0

    def _get_entity_sentiment(self, ticker):
        """
        ⚡ 優化版：讀取 LLM 提取的精確個股實體情緒

        效能提升：使用快取管理器代替檔案系統搜尋
        """
        try:
            # 從快取管理器讀取實體數據
            data = self.cache_manager.get("entity_sentiment", ticker)

            if data:
                entities = data.get("entities", [])
                code = ticker.split(".")[0]
                for ent in entities:
                    if ent.get("ticker") == code:
                        return float(ent.get("sentiment", 0.0))
        except:
            pass

        # Fallback：嘗試舊方法（相容性）
        import json

        try:
            # 優先搜尋實體級別快取 (由 NewsAnalyzer.analyze_entities_with_llm 生成)
            entity_cache = os.path.join(
                self.sentiment_analyzer.cache_file.replace(
                    "market_sentiment_cache.json", "news_cache"
                ),
                f"entities_{ticker}.json",
            )
            if os.path.exists(entity_cache):
                with open(entity_cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entities = data.get("entities", [])
                    # 在實體列表中尋找匹配的 ticker
                    code = ticker.split(".")[0]
                    for ent in entities:
                        if ent.get("ticker") == code:
                            sentiment = float(ent.get("sentiment", 0.0))

                            # 自動遷移到新快取
                            self.cache_manager.set(
                                "entity_sentiment", ticker, data, ttl=7 * 24 * 3600
                            )

                            return sentiment
        except:
            pass
        return None

    def prepare_historical_tensor(self, tickers, days=250):
        """
        🚀 核心優化：優先從 Parquet 讀取快照數據
        """
        print(f"📊 正在準備歷史時空數據 (範圍: {days} 天)...")

        # 確保快照存在 (調用通用 MarketDataManager 功能)
        self.data_mgr.maintain_categorized_snapshots()

        market_sent = self.sentiment_analyzer.get_market_sentiment()
        market_score = market_sent.get("score", 50)

        df_stocks = get_tw_stock_list()
        # 建立 Ticker 到產業的映射
        ticker_to_industry = dict(zip(df_stocks["Ticker"], df_stocks["產業"]))

        # 判斷需要加載哪些產業的 Parquet
        target_industries = set([ticker_to_industry.get(t, "其他") for t in tickers])

        # 預先加載產業 Parquet 到記憶體
        industry_cache = {}
        for ind in target_industries:
            ind_df = self.data_mgr.load_industry_snapshot(ind)
            if not ind_df.empty:
                industry_cache[ind] = ind_df

        stock_dfs = {}
        all_dates = None

        for ticker in tickers:
            try:
                # 優先從產業快照提取
                industry = ticker_to_industry.get(ticker, "其他")
                if industry in industry_cache:
                    full_ind_df = industry_cache[industry]
                    df = full_ind_df[full_ind_df["Ticker"] == ticker].copy()
                else:
                    # Fallback 到原始 CSV
                    _, df, _, _ = get_stock_data(ticker)

                if len(df) < days + self.lookback:
                    continue

                df = tech_analysis.calculate_technical_indicators(df, ticker=ticker)

                # --- [旗艦演算法升級] 多層級新聞情緒融合 ---
                # 1. 基礎聲量代理
                df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
                df["Internet_Attention"] = (
                    df["Volume"] / (df["Volume_MA20"] + 1e-8)
                ).fillna(1.0)

                # 2. 情緒特徵初始化 (預設使用 K 線動能代理)
                df["K_Line_Body"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-8)
                df["Sentiment_Proxy"] = (
                    df["K_Line_Body"] * df["Internet_Attention"] * 10
                ).fillna(0.0)

                # 3. 注入 LLM 提取的「實體級別」精確分數 (最優先)
                entity_score = self._get_entity_sentiment(ticker)
                if entity_score is not None:
                    # 將 LLM 實體分數覆蓋到最後一筆數據，影響明日預測
                    df.iloc[-1, df.columns.get_loc("Sentiment_Proxy")] = (
                        entity_score * 5.0
                    )  # 加權處理
                    df.iloc[
                        -1, df.columns.get_loc("Internet_Attention")
                    ] += 2.0  # 增加關注度權重
                else:
                    # 4. 次優先：注入通用新聞情緒
                    real_score, news_count = self._get_latest_news_sentiment(ticker)
                    if news_count > 0:
                        df.iloc[-1, df.columns.get_loc("Sentiment_Proxy")] = real_score
                        df.iloc[-1, df.columns.get_loc("Internet_Attention")] = (
                            news_count * 1.5
                        )

                if "Market_Sentiment_Score" not in df.columns:
                    df["Market_Sentiment_Score"] = market_score

                df = df.tail(days + self.lookback)
                stock_dfs[ticker] = df
                if all_dates is None:
                    all_dates = df.index
                else:
                    all_dates = all_dates.intersection(df.index)
            except:
                continue

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
                        future_ret = (df["Close"].iloc[next_idx] - row["Close"]) / row[
                            "Close"
                        ]
                        labels_tensor[i, j] = future_ret
                except:
                    pass

        return data_tensor, labels_tensor, valid_tickers

    def train_market_gnn(self, tickers=None, epochs=50, progress_callback=None):
        """
        使用滑動窗口訓練 ST-GNN
        """
        if tickers is None or len(tickers) == 0:
            return "❌ 錯誤：未提供訓練標的名單"

        # 1. 準備數據
        if progress_callback:
            progress_callback(f"📊 正在準備 {len(tickers)} 支股票的歷史時空數據與圖結構...", 0.1)
        raw_data, raw_labels, valid_tickers = self.prepare_historical_tensor(tickers)
        adj = self.build_market_graph(valid_tickers)

        # 2. 構建滑動窗口樣本
        if progress_callback:
            progress_callback("🧩 正在構建滑動窗口樣本...", 0.3)
        # X: [Samples, Seq_Len, Nodes, Features]
        # Y: [Samples, Nodes, 1]
        X, Y = [], []
        for i in range(len(raw_data) - self.lookback - 1):
            X.append(raw_data[i : i + self.lookback])
            Y.append(raw_labels[i + self.lookback])

        X = torch.from_numpy(np.array(X)).float().to(self.device)
        Y = torch.from_numpy(np.array(Y)).float().to(self.device).unsqueeze(-1)

        if progress_callback:
            progress_callback(f"🔥 訓練樣本構建完成: {X.shape}", 0.4)
        print(f"🔥 訓練樣本構建完成: {X.shape}")

        # 3. 訓練
        model = SpatioTemporalGNN(
            node_features=len(self.feature_columns),
            hidden_dim=128,
            seq_len=self.lookback,
            output_dim=1,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.HuberLoss()

        model.train()
        for epoch in range(epochs):
            # 這裡我們使用全量訓練 (如果 VRAM 不夠需改用 DataLoader)
            optimizer.zero_grad()
            output = model(X, adj)  # [Samples, Nodes, 1]
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}"
                print(msg)
                if progress_callback:
                    prog = 0.4 + (0.5 * (epoch + 1) / epochs)
                    progress_callback(msg, prog)

        # 4. 保存
        if progress_callback:
            progress_callback("💾 正在保存 GNN 模型...", 0.95)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tickers": valid_tickers,
                "features": self.feature_columns,
                "lookback": self.lookback,
            },
            self.model_path,
        )

        return f"✅ ST-GNN 歷史訓練完成！涵蓋 {len(valid_tickers)} 支股票 x {len(X)} 個時間樣本。"

    def build_market_graph(self, tickers, data_tensor=None):
        """
        🚀 演算法升級：構建融合「靜態產業」與「動態網路情緒」的知識圖譜
        :param data_tensor: 形狀為 [Time, Nodes, Features] 的張量
        """
        num_nodes = len(tickers)
        adj = np.eye(num_nodes)
        df_stocks = get_tw_stock_list()

        # 1. 靜態圖：產業連動 (基礎共識)
        industry_map = {}
        for ticker in tickers:
            code = ticker.split(".")[0]
            row = df_stocks[df_stocks["代碼"] == code]
            if not row.empty:
                ind = row.iloc[0]["產業"]
                if ind not in industry_map:
                    industry_map[ind] = []
                industry_map[ind].append(ticker)

        t_to_idx = {t: i for i, t in enumerate(tickers)}
        for ind, members in industry_map.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if members[i] in t_to_idx and members[j] in t_to_idx:
                        idx_i, idx_j = t_to_idx[members[i]], t_to_idx[members[j]]
                        adj[idx_i, idx_j] += 0.5
                        adj[idx_j, idx_i] += 0.5

        # 2. 動態圖：市場走勢與隱藏網路關聯 (如：供應鏈、題材概念股)
        if data_tensor is not None and data_tensor.shape[0] > 1:
            try:
                # 提取 Close 價格 (假設在 feature index 0)
                close_prices = data_tensor[:, :, 0]  # [Time, Nodes]
                # 計算每日報酬率
                returns = np.diff(close_prices, axis=0) / (close_prices[:-1, :] + 1e-8)

                # 計算 Pearson 相關係數矩陣，找出走勢高度黏合的標的
                corr_matrix = np.corrcoef(returns, rowvar=False)  # [Nodes, Nodes]
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                # 將高相關性 (>0.6) 視為隱藏連動，建立動態邊
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        if corr_matrix[i, j] > 0.6:
                            weight = 0.5 * float(corr_matrix[i, j])
                            adj[i, j] += weight
                            adj[j, i] += weight
            except Exception as e:
                print(f"動態相關性圖構建失敗: {e}")

        # 正規化 (GCN 標準推導)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        norm_adj = np.diag(d_inv_sqrt).dot(adj).dot(np.diag(d_inv_sqrt))

        return torch.from_numpy(norm_adj).float().to(self.device)

    def prepare_node_features(self, tickers):
        # 為了相容性保留此方法，但在 v2.0 中主要使用 prepare_historical_tensor
        return None, None, tickers

    def predict_market_relations(self):
        """使用訓練好的 ST-GNN 預測明日強度"""
        if not os.path.exists(self.model_path):
            return "❌ 未訓練"
        checkpoint = torch.load(self.model_path, map_location=self.device)
        tickers = checkpoint["tickers"]
        saved_features = checkpoint.get("features", [])

        # ⚠️ 演算法升級防呆機制：檢查特徵數量是否一致
        if len(saved_features) != len(self.feature_columns):
            return {
                "error": -1,
                "message": "⚠️ 偵測到 GNN 演算法與特徵已升級（新增了網路情緒與聲量），請至「模型訓練中心」重新執行【全市場關係訓練】以套用最新演算法。",
            }

        # 準備最近一段時間的數據 (以利計算動態圖相關性)
        raw_data, _, valid_tickers = self.prepare_historical_tensor(tickers, days=30)

        # 構建融合圖譜
        adj = self.build_market_graph(valid_tickers, raw_data)

        # 取最後 lookback 天
        x_input = (
            torch.from_numpy(raw_data[-self.lookback :])
            .float()
            .to(self.device)
            .unsqueeze(0)
        )

        model = SpatioTemporalGNN(len(self.feature_columns), 128, self.lookback, 1).to(
            self.device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            preds = model(x_input, adj)

        return dict(zip(valid_tickers, preds.cpu().numpy().flatten()))
