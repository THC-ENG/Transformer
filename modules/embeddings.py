#此部分为Embeddings，将输入Token转换为向量
import torch
from torch import nn
import torch.nn.functional as F
import math


#测试
# random_torch = torch.rand(4,4)
# print(random_torch)

from torch import Tensor
#将输入词汇表索引转换成指定维度的embedding

class TokenEmbedding(nn.Embedding):     #词表索引
    def __init__(self, vocab_size, d_model):   #词汇表大小、维度
            super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


#通过位置编码，计算输入序列的每个词生成的正弦/余弦位置编码，使模型利用位置信息
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()    #继承父类

        encoding = torch.zeros(max_len, d_model, device=device)
        encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)    #定义序列
        pos = pos.float().unsqueeze(dim=1)    #转化为二维张量，表示词的位置
        _2i = torch.arange(0, d_model, step=2, device=device).float()    #生成序列，用来区分偶数/奇数维度

        encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))    #计算位置编码,偶数
        encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))    #奇数  -->  相邻位置相似，远的位置差异大，可以算出来谁离谁近
        
        self.register_buffer('encoding', encoding)    # model.to(device)时张量会自动跟去gpu
    
    def forward(self, x):    #前向传播过程
        batch_size, seq_len = x.size()    #获取输入x的大小
        return self.encoding[:seq_len,:]    #返回位置编码矩阵中前seq_len的行数，#即seq_len和d_model大小的子矩阵
        #应与前面的tokenembedding相加，得到位置信息，在TransformerEmbedding中实现


#两部分结合，得到最终的张量
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)    #随即丢弃神经元，防止过拟合
        self.d_model = d_model # 记录一下维度
    
    def forward(self, x):    #定义输入x
        tok_emb = self.tok_emb(x)    #计算两个embedding
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)    #返回tok_emb + pos_emb并应用drop_out层之后的变量