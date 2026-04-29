import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGNN(nn.Module):
    """時空圖神經網路 (ST-GNN) v2.0"""
    def __init__(self, node_features, hidden_dim, seq_len, output_dim):
        super(SpatioTemporalGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.gnn_layer = nn.Linear(node_features, hidden_dim)
        self.lstm = nn.LSTM(node_features + hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_seq, adj):
        batch_size, seq_len, num_nodes, _ = x_seq.shape
        gnn_out_seq = []
        for t in range(seq_len):
            xt = x_seq[:, t, :, :]
            support = self.gnn_layer(xt)
            gnn_h = torch.matmul(adj, support)
            gnn_out_seq.append(gnn_h.unsqueeze(1))
        gnn_context = torch.cat(gnn_out_seq, dim=1)
        combined = torch.cat([x_seq, gnn_context], dim=-1)
        combined = combined.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(combined)
        final_h = lstm_out[:, -1, :] 
        out = self.fc(final_h)
        return out.reshape(batch_size, num_nodes, -1)

class HybridTransLSTM(nn.Module):
    """🚀 2026 旗艦終極版：Deep Transformer-LSTM 混合主模型"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=6, nhead=16, seq_len=20, output_dim=1):
        super(HybridTransLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=0.15, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.2)
        self.fusion_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x_proj = self.norm1(self.input_projection(x))
        trans_out = self.transformer(x_proj) 
        lstm_out, _ = self.lstm(x_proj)
        fused_feat, _ = self.fusion_att(lstm_out[:, -1:, :], trans_out, trans_out)
        return self.fc(fused_feat.squeeze(1))

class LSTMModel(nn.Module):
    """殘差注意力 LSTM"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.residual_layer = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        attn_weights = self.attention(lstm_out) 
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        residual = self.residual_layer(x[:, -1, :])
        return self.fc(context + residual)
