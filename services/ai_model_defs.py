import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 優化 1: 增加 Dropout (0.3) 防止過擬合
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        # 優化 2: 增加感知機層 (MLP Head) 提取非線性特徵，而非直接輸出
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # 確保初始隱藏狀態也在同一個裝置上
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # 取序列最後一個時間點的輸出
        out = out[:, -1, :] 
        
        # 通過 MLP 層
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    """
    GRU 模型 (Gated Recurrent Unit)
    優化版：加入 Batch Normalization、Layer Normalization 和殘差連接，提升訓練穩定性和準確度
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 輸入正規化層（穩定訓練）
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # GRU 結構（雙向 + Dropout）
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=False  # 單向以保持預測方向性
        )
        
        # Layer Normalization（比 BatchNorm 更適合序列數據）
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 增強 MLP Head（更深層次的特徵提取）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # 輸入正規化（對每個特徵維度）
        # x shape: (batch, seq_len, features)
        x_norm = x.transpose(1, 2)  # (batch, features, seq_len)
        x_norm = self.input_norm(x_norm)
        x_norm = x_norm.transpose(1, 2)  # (batch, seq_len, features)
        
        # GRU 處理
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        out, _ = self.gru(x_norm, h0)
        
        # 取最後時間步 + Layer Norm
        out = out[:, -1, :]
        out = self.layer_norm(out)
        
        # 殘差連接輔助變量（儲存中間層輸出）
        identity = out
        
        # MLP Head with Residual Connection
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 殘差連接（跳過第一層）
        out = out + identity
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out


class TransformerModel(nn.Module):
    """
    🆕 Transformer 模型 - v5.0 新增
    優勢：更強的長期依賴捕捉能力，平行處理所有時間步
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, nhead=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_type = 'Transformer'
        
        # 輸入嵌入層
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出 MLP Head
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Embedding
        x = self.input_embedding(x)  # (batch, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        
        # Transformer Encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden_dim)
        
        # 取最後時間步輸出
        x = x[:, -1, :]  # (batch, hidden_dim)
        x = self.layer_norm(x)
        
        # MLP Head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """位置編碼，為 Transformer 提供序列位置資訊"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 創建位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionLSTM(nn.Module):
    """
    🆕 LSTM + Attention 模型 - v5.0 新增
    優勢：自動識別哪些時間點最重要
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention 機制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP Head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # LSTM 處理
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_dim)
        
        # 殘差連接 + Layer Norm
        lstm_out = self.layer_norm1(lstm_out)
        
        # Self-Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 殘差連接
        out = lstm_out + attn_out
        out = self.layer_norm2(out)
        
        # 取最後時間步
        out = out[:, -1, :]
        
        # MLP Head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class EnsembleModel(nn.Module):
    """
    🆕 模型組合系統 - v5.0 新增
    整合 LSTM + GRU + Transformer，投票決策
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(EnsembleModel, self).__init__()
        self.model_type = 'Ensemble'
        
        # 三個子模型
        self.lstm = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        self.gru = GRUModel(input_dim, hidden_dim, num_layers, output_dim)
        self.transformer = TransformerModel(input_dim, hidden_dim, num_layers, output_dim)
        
        # 動態權重（可學習）
        self.weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.4]))  # LSTM, GRU, Transformer
        
    def forward(self, x):
        # 三個模型的預測
        lstm_pred = self.lstm(x)
        gru_pred = self.gru(x)
        transformer_pred = self.transformer(x)
        
        # Softmax 權重（確保加總為1）
        weights = torch.softmax(self.weights, dim=0)
        
        # 加權平均
        ensemble_pred = (
            weights[0] * lstm_pred +
            weights[1] * gru_pred +
            weights[2] * transformer_pred
        )
        
        return ensemble_pred
    
    def get_weights(self):
        """獲取當前權重分配"""
        weights = torch.softmax(self.weights, dim=0)
        return {
            'LSTM': weights[0].item(),
            'GRU': weights[1].item(),
            'Transformer': weights[2].item()
        }
