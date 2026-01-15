#实现层归一化操作
import torch
from torch import nn
import math


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):               # d_model为词向量的维度
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))     #可训练缩放参数γ，初始化为全1张量
        self.beta = nn.Parameter(torch.zeros(d_model))     #可训练偏移参数β，初始化为0
        self.eps = eps                                     #加一个极小数防止除以0

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)     #计算x在最后一个维度上的均值,并保持维度不变
        var = x.var(-1, unbiased = False, keepdim = True)      #计算x在最后一个维度上的方差,不使用无偏估计
        
        out = (x - mean)/torch.sqrt(var + self.eps)     #对x归一化
        out = self.gamma*out + self.beta     #缩放平移得到输出
        
        return out
    

#LayerNorm 与 BatchNorm 区别

#LayerNorm 的归一化对象是每个词，即一个 sample 的所有 feature ，与 batch size 无关；
#而 BatchNorm 归一化对象是整个 batch 在一个 feature 上的表现，不适用于 NLP.'''