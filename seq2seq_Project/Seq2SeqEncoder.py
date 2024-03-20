import torch
from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # 初始化 GRU; input_size和hidden_size参数统一通过hidden_size设置
        # 因为输入张量的大小就是Embedding层的特征数量 features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True, batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # 词索引对应Embedding层的向量索引
        embedded = self.embedding(input_seq)
        # 以批次为单位，把用0填充的，不同长度的词向量序列转换为一维向量和长度向量
        # 为后面的RNN计算打包训练数据
        # 如果词向量序列没有排序，设置enforce_sorted=False。如果已经过排序，enforce_sorted=True
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        # 打包后的数据传递给 GRU
        _, hidden_states = self.gru(packed, hidden)
        # 拼接最后输出的hidden state
        states = hidden_states.split(1,dim=0)  # ([1,batch_size,state_size], [1,batch_size,state_size])
        out_states = torch.cat(states,dim=-1)  # [1,batch_size,state_size*2]
        # out_states = torch.add(states[0],states[1])
        # out_states = torch.mul(states[0], states[1])
        return out_states

if __name__ == '__main__':
    import os
    from CorpuProcess import Seq2SeqDataset,read_corpus, build_dataloader

    batch_size = 4
    emb_hidden_size = 10
    rnn_hidden_size = 10
    encoder_n_layers = 1

    data_file = os.path.join(os.path.dirname(__file__), 'dataset.txt')
    encode_datas,decode_datas = read_corpus(data_file)
    dataset = Seq2SeqDataset(encode_datas, decode_datas)
    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)

    embedding = nn.Embedding(len(dataset.encode_vocab), emb_hidden_size, padding_idx=0)
    encoder = EncoderRNN(rnn_hidden_size, embedding, encoder_n_layers)
    for enc,enc_len,dec,dec_mask in dataloader:
        print(enc.shape)
        hidden_states = encoder(enc, enc_len)
        print(hidden_states.shape)
        break
    