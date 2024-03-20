import torch
from torch import nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size,embedding, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # 保留属性
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义层
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden):
        # 每次运行，我们只获取一个词
        # 获取当前输入层的词向量
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 单向的GRU
        rnn_output, last_hidden = self.gru(embedded, last_hidden)
        # 由于是逐token处理，所以sequence_length始终为1。直接降维
        output = rnn_output.squeeze(1)
        # 输出
        output = self.out(output)
        # 求取最大概率
        output = F.softmax(output, dim=1)
        # 返回output和RNN最后的hidden state
        return output, last_hidden

if __name__ == '__main__':
    import os
    from Seq2SeqEncoder import EncoderRNN
    from CorpuProcess import Seq2SeqDataset,read_corpus, build_dataloader

    batch_size = 32
    emb_hidden_size = 10
    rnn_hidden_size = 10
    encoder_n_layers = 1
    decoder_n_layers = 1

    data_file = os.path.join(os.path.dirname(__file__), 'dataset.txt')
    encode_datas,decode_datas = read_corpus(data_file)
    dataset = Seq2SeqDataset(encode_datas, decode_datas)
    dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)

    encoder_embedding = nn.Embedding(len(dataset.encode_vocab), emb_hidden_size, padding_idx=0)
    decoder_embedding = nn.Embedding(len(dataset.decode_vocab), emb_hidden_size * 2, padding_idx=0)
    encoder = EncoderRNN(rnn_hidden_size, encoder_embedding, encoder_n_layers)
    decoder = DecoderRNN(rnn_hidden_size * 2, decoder_embedding, len(dataset.decode_vocab),decoder_n_layers)

    num = 0
    for enc,enc_len,dec,dec_mask in dataloader:
        hidden_states = encoder(enc,enc_len)
        # 解码输入序列的长度
        dec_seq_length = dec.size(1)
        # 解码输入第一个token:"SOS"
        decode_input = dec[:,0].reshape(-1,1)
        # 解码
        output, hidden_state = decoder(decode_input, hidden_states)
        print(decode_input.shape) # [batch_size,1]
        print(output.shape) # [batch_size, 1, vocab_size]
        print(hidden_states.shape) # [1, batch_size, state_size]
        break
