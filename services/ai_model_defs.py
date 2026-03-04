import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGNN(nn.Module):
    """
    🚀 時空圖神經網路 (ST-GNN) v2.0
    結合 GNN (空間/關係) 與 LSTM (時間/序列)
    """
    def __init__(self, node_features, hidden_dim, seq_len, output_dim):
        super(SpatioTemporalGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 1. 空間特徵提取 (GNN)
        self.gnn_layer = nn.Linear(node_features, hidden_dim)
        
        # 2. 時間特徵提取 (LSTM)
        # 輸入是原始特徵 + GNN 提取的關聯特徵
        self.lstm = nn.LSTM(node_features + hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        
        # 3. 輸出層
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_seq, adj):
        """
        x_seq: [batch, seq_len, num_nodes, node_features]
        adj: [num_nodes, num_nodes]
        """
        batch_size, seq_len, num_nodes, _ = x_seq.shape
        
        # 空間卷積
        gnn_out_seq = []
        for t in range(seq_len):
            xt = x_seq[:, t, :, :] # [batch, num_nodes, features]
            support = self.gnn_layer(xt) # [batch, num_nodes, hidden]
            gnn_h = torch.matmul(adj, support) # [batch, num_nodes, hidden]
            gnn_out_seq.append(gnn_h.unsqueeze(1))
            
        gnn_context = torch.cat(gnn_out_seq, dim=1) # [batch, seq_len, num_nodes, hidden]
        
        # 融合空間與時間特徵
        combined = torch.cat([x_seq, gnn_context], dim=-1) # [batch, seq_len, num_nodes, features + hidden]
        
        # 針對每一支股票處理序列
        combined = combined.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, -1)
        
        lstm_out, _ = self.lstm(combined)
        final_h = lstm_out[:, -1, :] 
        
        out = self.fc(final_h)
        return out.reshape(batch_size, num_nodes, -1)

class LSTMModel(nn.Module):
    """
    優化版殘差注意力 LSTM (Residual Attention LSTM)
    適合作為主模型 (Base Model) 提取通用市場特徵
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多層 LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # 注意力機制層
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 殘差連接維度對齊
        self.residual_layer = nn.Linear(input_dim, hidden_dim)
        
        # 輸出 MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x) 
        
        # Attention 機制
        attn_weights = self.attention(lstm_out) 
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        
        # 殘差連接
        residual = self.residual_layer(x[:, -1, :])
        combined = context + residual
        
        return self.fc(combined)

class GNNModel(nn.Module):
    """
    保留基礎 GNN 模型定義以維持向下相容性
    """
    def __init__(self, node_features, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.gc1 = nn.Linear(node_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(torch.matmul(adj, x)))
        return self.fc(x)
