import torch
from torch import nn
import torch.nn.functional as F
import math

from modules.embeddings import TransformerEmbedding
from modules.attention import MultiHeadAttention
from modules.layernorm import LayerNorm



# 首先是 FeedForward 模块(Position-wise Feed-Forward Network)
# FFN 对序列中的每个 Token 是独立进行处理的，不看上下文，只关注当前位置的非线性特征变换
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):    #正则化防止过拟合
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)     #第一层线性映射，全连接层，输入 d_model, 输出 hidden，升维
        self.fc2 = nn.Linear(hidden, d_model)     #第二层线性映射（降维），把维度拉回 d_model
        self.dropout = nn.Dropout(dropout)        # dropout 层正则化
        
    def forward(self, x):
        x = self.fc1(x)        #将输入的 x 传入给fc1
        x = F.relu(x)          #激活函数Relu
        x = self.dropout(x)
        x = self.fc2(x)        #防止过拟合，映射回 d_model 维度
        return x
    


#然后编写 Encoder Layer (Add & Norm)，结构如下：
    # Input 
    #  ├─ Multi-Head Self-Attention 
    #  ├─ Add & Norm 
    #  ├─ Feed Forward 
    #  └─ Add & Norm 

    # encoderLayer 的输入输出: 
    # 输入:  (batch, seq_len, d_model) 
    # 输出:  (batch, seq_len, d_model)

#这里实现的是 Post-LN（先加后归一化）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)   #定义多头注意力
        self.norm1 = LayerNorm(d_model)         #定义归一化层
        self.dropout1 = nn.Dropout(dropout)     #定义 dropout 层
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)      #定义前馈神经网络
        self.norm2 = LayerNorm(d_model)         #两个子层各自一个 LayerNorm，不能共用
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #第一子层：Self-Attention ：LayerNorm(x+Attention(x))
        _x = x      #保存原始输入，便于使用 
        x = self.attention(x, x, x, mask)       # q,k,v分别通过 x 传入，mask 考虑忽略某些位置
        x = self.dropout1(x)     #前向传播
        x = self.norm1(x + _x)   #残差连接，归一化，避免了梯度消失
        #第二子层：FFN ：LayerNorm(x+FFN(x))
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x) 
        return x
    


#Encoder 包含多个 Encoder Layer，实现了多层 transformer 编码器
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)        #定义多个 Encoder Layer 的实例
            ]
        )
    
    def forward(self, x, mask):
        x = self.embedding(x)      #通过词汇表索引映射到高维向量空间，完成 Token embedding, Positional encoding, Dropout
        for layer in self.layers:         #通过编码器前向传播
            x = layer(x, mask)            # Transformer Encoder 堆叠方式
        return x