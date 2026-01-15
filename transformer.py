import torch
from torch import nn
import torch.nn.functional as F
import math

from modules.embeddings import TransformerEmbedding
from modules.attention import MultiHeadAttention
from modules.layernorm import LayerNorm
from encoder import PositionwiseFeedForward
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, 
                 src_pad_idx,    # padding token 的索引，用于构造 mask
                 trg_pad_idx,    # 防止 padding 位置参与 attention

                 enc_voc_size,   # 源语言 / 目标语言词表大小
                 dec_voc_size,   # 决定 embedding 层大小、decoder 输出维度

                 d_model,  # Transformer 的核心维度：embedding 维度、attention 输入输出维度、residual 通道维度
                 max_len,  # 位置编码最大长度，决定模型能处理的最长序列
                 n_head,   # Multi-Head Attention 的头数
                 ffn_hidden, # FFN 中间层维度
                 n_layer,    # Encoder / Decoder 堆叠层数
                 drop_prob,  # Dropout 概率
                 device):    #CPU / GPU，用于 mask 和张量放置
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, 
                               max_len, 
                               d_model, 
                               ffn_hidden, 
                               n_head, 
                               n_layer, 
                               drop_prob, 
                               device)
        
        self.decoder = Decoder(dec_voc_size, 
                               max_len, 
                               d_model, 
                               ffn_hidden, 
                               n_head, 
                               n_layer, 
                               drop_prob, 
                               device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    #关于 head mask 
    #填充掩码 [Batch, 1, 1, Seq_Len]
    def make_pad_mask(self, x, pad_idx):         #填充部分不应对模型输出产生影响
        # x: [batch, seq_len]
        # output: [batch, 1, 1, seq_len]
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)  #把它变成 [batch, 1, 1, seq_len]
        return mask
    
    #因果掩码 [Seq_Len, Seq_Len] --> 仅用于 Decoder 的第一层 Self-Attention
    def make_causal_mask(self, len_q, len_k,):
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    # src: [batch, src_len]
    # trg: [batch, trg_len]
    def forward(self, src, trg):
        # 1. 生成 Encoder Mask (Source Mask)，让 Encoder 忽略掉 Padding 部分
        src_mask = self.make_pad_mask(src, self.src_pad_idx)      #为原序列创造填充掩码
        
        # 2. 生成 Decoder Mask (Target Mask). Decoder 需要两个限制：
        #   a. 不能看 Padding (pad_mask)
        #   b. 不能看未来 (causal_mask)
        trg_pad_mask = self.make_pad_mask(trg, self.trg_pad_idx) # [batch, 1, 1, trg_len]
        trg_causal_mask = self.make_causal_mask(trg.size(1), trg.size(1))    # [trg_len, trg_len]
       
        # [batch, 1, 1, trg_len] & [trg_len, trg_len] 
        # 自动变成 -> [batch, 1, trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_causal_mask
        
        enc = self.encoder(src, src_mask)      #原序列传入 encoder
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out