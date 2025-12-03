import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        # 截取对应长度的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerActor(nn.Module):
    def __init__(self, obs_size, act_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0):
        super(TransformerActor, self).__init__()
        
        # 1. 嵌入层：将观测维度映射到 Transformer 维度
        self.embedding = nn.Linear(obs_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 2. Transformer Encoder
        # batch_first=True 意味着输入格式为 (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 输出层
        self.output_head = nn.Linear(d_model, act_size)
        self.d_model = d_model

    def forward(self, x):
        # 兼容性处理：如果输入是 2D (Batch, Obs)，转为 3D (Batch, 1, Obs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 通过 Transformer
        x = self.transformer(x)
        
        # 取序列最后一个 token 作为输出特征 (对于 Seq=1，就是唯一的那个)
        x = x[:, -1, :]
        
        # 连续动作空间通常使用 Tanh 激活到 [-1, 1]
        return torch.tanh(self.output_head(x))

class TransformerCritic(nn.Module):
    def __init__(self, obs_size, act_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.0):
        super(TransformerCritic, self).__init__()
        
        # Critic 输入包含状态和动作
        input_dim = obs_size + act_size
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, obs, act):
        # 拼接 Observation 和 Action
        x = torch.cat([obs, act], dim=-1)
        
        # 兼容性处理：2D -> 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        x = self.transformer(x)
        
        # 取最后一个 token
        x = x[:, -1, :]
        
        return self.output_head(x)

# 映射类名，保持 agent.py 兼容
Actor = TransformerActor
Critic = TransformerCritic