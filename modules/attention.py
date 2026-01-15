import torch
from torch import nn
import math


#输入变量
x = torch.rand(128, 32, 512)    #三个维度batch,time,dimension
d_model = 512     #每个词的长度
n_head = 8      #头数


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):    # n_head为头数
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head       #实例变量
        self.d_model = d_model
        assert d_model % n_head == 0

        #定义Q,K,V
        self.w_q = nn.Linear(d_model, d_model)    #定义线性变换
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

        #前向传播计算
    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape     #定义计算
        n_d = self.d_model//self.n_head     #每个头的维度=64
        # 1. 线性投影
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)     #对q,k,v分别线性变换
        # 2. 分头
        len_k = k.size(1)
        len_v = v.size(1) # 通常 len_k == len_v
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)     # 重塑为多头形状，便于一次性并行算出8个头的注意力
        k = k.view(batch, len_k, self.n_head, n_d).permute(0, 2, 1, 3)     # view:把最后一维 512 切成 [8, 64]
        v = v.view(batch, len_v, self.n_head, n_d).permute(0, 2, 1, 3)     # permute:把头的维度换到前面去。
        
        score = q @ k.transpose(2, 3)/math.sqrt(n_d)     #计算注意力分数

        if mask is not None:     #掩码
            score = score.masked_fill(mask == 0, -1e9)   #把不想被看到的每一个位置设为一个很小的负数，便于注意力分数接近于0
        score = self.softmax(score) @ v    #归一化并取值

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)      #加权值重新排列
        out = self.w_combine(score)

        return out
    
#实例化
# attention = MultiHeadAttention(d_model, n_head)
# out = attention(x, x, x)

# print("Output shape:", out.shape)