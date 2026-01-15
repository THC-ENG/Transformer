import torch
import torch.nn as nn
import torch.optim as optim
import time
from transformer import Transformer
import torch.optim as optim
import random


# 1. 配置参数 (搞个迷你的配置，跑得快)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Running on: {device}")

# 定义一些超参数
enc_voc_size = 100  # 词表大小 (假设只有100个不同的字)
dec_voc_size = 100
d_model = 64        # 向量维度 (不用512，64够了)
ffn_hidden = 128    # FFN隐藏层
n_head = 4          # 4个头
n_layer = 2         # 2层
max_len = 50        # 最大长度
batch_size = 8
src_pad_idx = 0     # 假设 0 是 padding
trg_pad_idx = 0
drop_prob = 0.1



# 2. 实例化模型
model = Transformer(src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, 
                    d_model, max_len, n_head, ffn_hidden, n_layer, 
                    drop_prob, device).to(device)

# 定义优化器 (Adam 是 Transformer 的标配)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数 (忽略 padding 的预测)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)



# 3. 构造伪造数据 (随机整数)
# 任务：让模型学会 "复制" (源序列是什么，目标序列就是什么)
# src: [Batch, Seq_Len]
src = torch.randint(1, enc_voc_size, (batch_size, 10)).to(device) # 长度为10的随机序列
trg = src.clone() # 目标和源一样

print("\n 模型结构已加载，准备点火...")
print(f"输入示例: {src[0].cpu().numpy()}")



# 4. 训练循环 (Training Loop)
model.train() # 切换到训练模式

for epoch in range(100): # 跑 100 轮
    optimizer.zero_grad() # 梯度清零
    
    # --- 核心数据处理 (Teacher Forcing) ---
    # Decoder 的输入: <sos>, A, B, C (去掉最后一个)
    # Decoder 的目标: A, B, C, <eos> (去掉第一个)
    # 这里我们简单粗暴处理：
    trg_input = trg[:, :-1] # 输入给模型看的前 9 个字
    targets = trg[:, 1:]    # 模型应该预测出的后 9 个字
    
    # --- 前向传播 ---
    # 输出 shape: [batch, seq_len, dec_voc_size]
    output = model(src, trg_input)
    
    # --- 计算 Loss ---
    # output 需要变成二维 [batch * seq_len, vocab_size]
    # targets 需要变成一维 [batch * seq_len]
    output_reshape = output.contiguous().view(-1, dec_voc_size)
    targets_reshape = targets.contiguous().view(-1)
    
    loss = criterion(output_reshape, targets_reshape)
    
    # --- 反向传播 ---
    loss.backward()
    optimizer.step()
    
    # 每 10 轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1:02} | Loss: {loss.item():.5f}')



# 5. 简单验证 (Inference)
print("\n 验证一下效果:")
model.eval()
with torch.no_grad():
    # 再次跑一次前向传播
    output = model(src, trg[:, :-1])
    # 取概率最大的那个词的索引 (argmax)
    prediction = output.argmax(dim=-1)
    
    print(f"原句 (Ground Truth): \n{targets[0].cpu().numpy()}")
    print(f"预测 (Prediction):   \n{prediction[0].cpu().numpy()}")

    if torch.equal(targets[0], prediction[0]):
        print("\n 成功！模型学会了复制任务！")
    else:
        print("\n 还有点误差，可能需要多训练几轮。")



#构建字典和分词器
# 1. 准备一点简单的中文训练数据 (Source -> Target)
# 这里我们可以做一个简单的“对话机器人”
sentences = [
    ("你好", "你好呀"),
    ("你是谁", "我是你的模型"),
    ("你爱我吗", "由于某种原因我无法回答"),
    ("具身智能", "那是我的未来"),
    ("跑代码", "好的马上跑"),
    # 再加几条用来测试泛化（虽然数据少泛化会很差）
    ("测试", "测试成功"),
    ("结束", "再见"),
]

# 2. 构建词表 (Vocabulary)
# 把所有出现的字都拆开，去重
all_chars = set()
for src, trg in sentences:
    for char in src: all_chars.add(char)
    for char in trg: all_chars.add(char)

# 加上特殊符号
# <pad>: 填充, <sos>: start of sentence, <eos>: end of sentence, <unk>: unknown
special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
# 排序并建立索引
vocab = special_tokens + list(sorted(list(all_chars)))

# 建立映射表
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

print(f" 字典构建完成！词表大小: {len(vocab)}")
print(f"前10个词: {vocab[:10]}")

# 3. 定义两个转换函数
def encode(text, max_len=20):
    # 中文 -> 数字索引序列
    # 1. 加上 <sos>
    seq = [char2idx.get('<sos>')]
    # 2. 转换正文
    for char in text:
        seq.append(char2idx.get(char, char2idx['<unk>']))
    # 3. 加上 <eos>
    seq.append(char2idx.get('<eos>'))
    
    # 4. 填充 Padding
    if len(seq) < max_len:
        seq += [char2idx['<pad>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len] # 截断
        
    return seq

def decode(indices):
    # 数字索引序列 -> 中文
    tokens = []
    for idx in indices:
        idx = idx.item() # 转成 python int
        if idx == char2idx['<eos>']: break # 遇到结束符停止
        if idx == char2idx['<sos>']: continue # 跳过开始符
        if idx == char2idx['<pad>']: continue # 跳过填充
        tokens.append(idx2char[idx])
    return "".join(tokens)

# 测试一下分词器
print(f"\n 测试分词器:")
demo_txt = "你好"
demo_idx = encode(demo_txt)
print(f"原文: {demo_txt}")
print(f"编码: {demo_idx}")
print(f"解码: {decode(torch.tensor(demo_idx))}")






#配置并训练模型
# ================= 配置参数 =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_pad_idx = char2idx['<pad>']
trg_pad_idx = char2idx['<pad>']
enc_voc_size = len(vocab) # 关键：词表大小要匹配
dec_voc_size = len(vocab)
d_model = 128    # 稍微大一点
n_head = 4
n_layer = 2
ffn_hidden = 256
max_len = 20     # 我们设定的最大长度
batch_size = len(sentences) # 全量训练（因为数据很少）
drop_prob = 0.1

# ================= 实例化模型 =================
model = Transformer(src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, 
                    d_model, max_len, n_head, ffn_hidden, n_layer, 
                    drop_prob, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

# ================= 准备 Batch 数据 =================
# 把所有句子都转成 Tensor
src_batch = []
trg_batch = []

for s, t in sentences:
    src_batch.append(encode(s, max_len))
    trg_batch.append(encode(t, max_len))

src_tensor = torch.LongTensor(src_batch).to(device) # [batch, len]
trg_tensor = torch.LongTensor(trg_batch).to(device) # [batch, len]

print("\n 开始训练中文 Transformer...")

# ================= 训练循环 =================
model.train()
for epoch in range(300): # 多跑几轮，因为是过拟合训练
    optimizer.zero_grad()
    
    # 构造输入和目标
    trg_input = trg_tensor[:, :-1]
    targets = trg_tensor[:, 1:]
    
    output = model(src_tensor, trg_input)
    
    # Reshape 计算 Loss
    output_reshape = output.contiguous().view(-1, dec_voc_size)
    targets_reshape = targets.contiguous().view(-1)
    
    loss = criterion(output_reshape, targets_reshape)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch: {epoch + 1} | Loss: {loss.item():.5f}')

# ================= 验证效果 =================
print("\n 见证奇迹的时刻:")
model.eval()

# 我们来测试这几句话
test_sentences = ["你好", "你是谁", "具身智能"]

with torch.no_grad():
    for txt in test_sentences:
        # 1. 准备输入
        src_idxs = torch.LongTensor([encode(txt, max_len)]).to(device)
        
        # 2. 生成 (贪婪搜索 Greedy Decoding)
        # 这里不能一次性给 trg，因为实际预测时我们不知道答案。
        # 我们必须一个字一个字地生成。
        
        # 初始输入只有 <sos>
        trg_idxs = torch.LongTensor([[char2idx['<sos>']]]).to(device)
        
        # 循环生成，直到遇到 <eos> 或达到最大长度
        for i in range(max_len):
            # 获取当前预测
            output = model(src_idxs, trg_idxs)
            # 取最后一个时间步的预测结果
            last_token_logits = output[:, -1, :]
            pred_token = last_token_logits.argmax(dim=-1).unsqueeze(1)
            
            # 把预测出来的字拼接到输入后面，作为下一次的输入
            trg_idxs = torch.cat((trg_idxs, pred_token), dim=1)
            
            # 如果预测出了 <eos>，就结束
            if pred_token.item() == char2idx['<eos>']:
                break
                
        # 解码并打印
        result = decode(trg_idxs[0])
        print(f"问: {txt} \t--->\t 答: {result}")


# 这些是它没见过的（不在 sentences 列表里）
unseen_sentences = [
    "你爱我",      # 训练集是 "你爱我吗"
    "谁是你",      # 训练集是 "你是谁"
    "你好不好",    # 训练集是 "你好"
]

print("\n 泛化能力测试 (它没背过的题):")
# ... (使用同样的验证代码运行这几个句子) ...

with torch.no_grad():
    for txt in unseen_sentences:
        # 1. 准备输入
        src_idxs = torch.LongTensor([encode(txt, max_len)]).to(device)
        
        # 2. 生成 (贪婪搜索 Greedy Decoding)
        # 这里不能一次性给 trg，因为实际预测时我们不知道答案。
        # 我们必须一个字一个字地生成。
        
        # 初始输入只有 <sos>
        trg_idxs = torch.LongTensor([[char2idx['<sos>']]]).to(device)
        
        # 循环生成，直到遇到 <eos> 或达到最大长度
        for i in range(max_len):
            # 获取当前预测
            output = model(src_idxs, trg_idxs)
            # 取最后一个时间步的预测结果
            last_token_logits = output[:, -1, :]
            pred_token = last_token_logits.argmax(dim=-1).unsqueeze(1)
            
            # 把预测出来的字拼接到输入后面，作为下一次的输入
            trg_idxs = torch.cat((trg_idxs, pred_token), dim=1)
            
            # 如果预测出了 <eos>，就结束
            if pred_token.item() == char2idx['<eos>']:
                break
                
        # 解码并打印
        result = decode(trg_idxs[0])
        print(f"问: {txt} \t--->\t 答: {result}")