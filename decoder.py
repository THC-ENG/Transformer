import torch
from torch import nn
import torch.nn.functional as F
import math

from modules.embeddings import TransformerEmbedding
from modules.attention import MultiHeadAttention
from modules.layernorm import LayerNorm
from encoder import PositionwiseFeedForward



#先编写 Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        # 1. 自注意力 (Masked Self-Attention)
        self.attention1 = MultiHeadAttention(d_model, n_head)     #创建多头注意力机制
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        # 2. 交叉注意力 (Cross-Attention)
        self.cross_attention = MultiHeadAttention(d_model, n_head)      #创建另一个多头注意力机制实例，即跨模态注意力机制，将 encoder 编码后的输入
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        # 3. 前馈网络 (FFN)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)

        self.dropout3 = nn.Dropout(drop_prob)


    def forward(self, dec, enc, t_mask, s_mask):
        # dec: Decoder 输入
        # enc: Encoder 输出 (Key, Value 的来源)

        # mask 说明：
        #Target Mask（t_mask）(Look-ahead mask, 遮住未来)
            #用于 Masked Self-Attention
            #阻止看到未来 token（下三角）
        #Source Mask（s_mask）(Padding mask, 遮住 Encoder 的填充符)
            # 用于 Cross-Attention
            # 屏蔽 padding token

        #第一层
        _x = dec       #保存原始解码器输入
        x = self.attention1(dec, dec, dec, t_mask)         #执行解码自注意力机制
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        #第二层，交叉注意力
        #数据流从 Decoder 跨越到了 Encoder，这是两个世界交汇的地方
        _x = x         #保存残差   
        # Q 来自 Decoder(x), K, V 来自 Encoder(enc)
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
       
        #第三层
        #把抓取到的原文信息和上文信息融合
        _x = x 
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x
    

#然后编整个 Decoder
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, dropout, device)      #创造 embedding 实例，将解码器输入词汇表转化为向量

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)        #定义多个 Decoder Layer 的实例
            ]
        )
    
        #定义全连接层，将解码器输出映射回解码器词汇表大小
        self.fc = nn.Linear(d_model, dec_voc_size)


    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)         #将编码器输出输入到 embedding 转化成向量

        for layer in self.layers:         #通过解码器前向传播
            dec = layer(dec, enc, t_mask, s_mask)  #堆叠的 Decoder Layers
        
        dec = self.fc(dec)            #经过全连接层映射回词表

        return dec