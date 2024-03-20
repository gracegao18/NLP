import os
import torch
import random

import os
from tqdm import tqdm
from torch import nn
from torch import optim
from Seq2SeqEncoder import EncoderRNN
from Seq2SeqDecoder import DecoderRNN
from CorpuProcess import Seq2SeqDataset,read_corpus, build_dataloader, corpus_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 强制训练率
teacher_forcing_ratio = 1.0
# 训练的轮数
epochs = 5
# 批次大小
batch_size = 32 
# 嵌入层隐藏层大小
emb_hidden_size = 64
# RNN隐藏层大小
rnn_hidden_size = 64
# 编码器层数
encoder_n_layers = 1
# 解码器层数
decoder_n_layers = 1
# 学习率
learning_rate = 0.0001
# 解码器学习率 = learning_rate * decoder_learning_ratio
decoder_learning_ratio = 5.0
# 训练时的间隔输出频次
print_every = 10

# 模型存盘名
model_name = 'sq_model'
# 存盘路径(存盘目录一部分)
save_dir = os.path.join("data", "save")
# 语料名(存盘目录一部分)
corpus_name = "numeric convert corpus"

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.reshape(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(encode_variable, lengths, decode_variable, mask, encoder, decoder,
          encoder_optimizer, decoder_optimizer):
    """encoder和decoder一个batch的训练"""

    # 梯度清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 训练设备
    input_variable = encode_variable.to(device)
    target_variable = decode_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # 前向传给编码器
    # 编码器最后一层的输出就是解码器的初始hidden state
    decoder_hidden = encoder(input_variable, lengths)

    # 创建初始化的解码器输入(每个语句起始token都是SOS)
    decoder_input = target_variable[:,0].reshape(-1,1)
    decoder_input = decoder_input.to(device)

    # 解码器序列长度
    max_target_len = target_variable.size(1)

    # 确认是否在迭代中使用强制teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 一次一个时间步通过解码器转发一批序列
    if use_teacher_forcing:
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            # teacher forcing: 下一个输入是正确的目标
            decoder_input = target_variable[:,t].reshape(-1,1)
            # 计算并累积损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:,t], mask[:,t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(1, max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            # 下一个输入是decoder自己的输出
            _, topi = decoder_output.topk(1)
            decoder_input = topi
            decoder_input = decoder_input.to(device)
            # 计算并累积损失
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:,t], mask[:,t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 后向传播
    loss.backward()

    # 更新模型梯度
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIter(dataloader, encoder, decoder, max_length):
    """模型训练"""

    # 模型优化器
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # 初始化
    print('初始化 ...')
    print_loss = 0

    # 循环迭代的次数 = 记录数 / batch_size
    n_iteration = len(dataloader)

    print("模型训练...")
    for epoch in range(epochs):
        pbar = tqdm(total=n_iteration)
        for iteration, (enc,enc_len,dec,dec_mask) in enumerate(dataloader):
            # 训练批次数据
            loss = train(enc, enc_len, dec, dec_mask, encoder,
                            decoder, encoder_optimizer, decoder_optimizer)
            print_loss += loss

            # 打印进程
            if iteration % print_every == print_every - 1:
                print_loss_avg = print_loss / print_every
                pbar.update(print_every)
                pbar.set_description("Epoc: {:-2d} Average loss: {:.5f}".format(epoch+1, print_loss_avg))
                print_loss = 0
        pbar.close()

        # 保存模型
        directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, rnn_hidden_size))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epochs': epochs,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'enc_voc_dict': dataset.encode_vocab.__dict__,
            'dec_voc_dict': dataset.decode_vocab.__dict__,
            'enc_embedding': encoder_embedding.state_dict(),
            'dec_embedding': decoder_embedding.state_dict(),
            'dec_max_length': max_length 
        }, os.path.join(directory, 'epoch_{}_{}.tar'.format(epoch+1, 'checkpoint')))
        

if __name__=='__main__':
    # 加载训练语料
    encode_datas,decode_datas = read_corpus(corpus_file)
    # 解码内容最大长度
    decode_max_length = max([len(d) for d in decode_datas]) + 1
    # 自定义数据集
    dataset = Seq2SeqDataset(encode_datas, decode_datas)
    # 模型训练用DataLoader
    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)

    dec_vocab = dataset.decode_vocab
    # 创建模型对象的Embedding
    encoder_embedding = nn.Embedding(len(dataset.encode_vocab), emb_hidden_size, padding_idx=0)
    decoder_embedding = nn.Embedding(len(dataset.decode_vocab), emb_hidden_size * 2, padding_idx=0)
    # 编码器、解码器对象
    encoder = EncoderRNN(rnn_hidden_size, encoder_embedding, encoder_n_layers)
    decoder = DecoderRNN(rnn_hidden_size * 2, decoder_embedding, len(dataset.decode_vocab),decoder_n_layers)
    encoder.to(device)
    decoder.to(device)
    # 训练并保存模型
    trainIter(dataloader, encoder, decoder, decode_max_length)