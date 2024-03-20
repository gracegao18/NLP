import os
import torch
import torch.nn as nn

from Seq2SeqEncoder import EncoderRNN
from Seq2SeqDecoder import DecoderRNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型配置参数
model_name = 'sq_model'
emb_hidden_size = 64
rnn_hidden_size = 64
encoder_n_layers = 1
decoder_n_layers = 1
dropout = 0.1
batch_size = 32

class Inference(nn.Module):
    def __init__(self, encoder, decoder):
        super(Inference, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # 输入token序列给encoder model
        encoder_hidden = self.encoder(input_seq, input_length)
        # 编码器最后一层的输入设置为解码器的初始hidden state
        decoder_hidden = encoder_hidden
        # 初始化解码的第一个token为decoder的SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * dec_voc['vocab']['SOS']
        # 初始化2个张量用于保存解码后的最佳token和得分
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # 一次迭代解码一个token
        for _ in range(max_length):
            # 当前解码的token传入decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # 获取最有可能的token标记及其softmax得分
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # 记录上面的token和得分
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # 当前decoder输出的token会作为下一次decoder的输入(添加一个维度)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # 返回收集的最匹配token列表和它们的得分
        return all_tokens, all_scores

def indexesFromSentence(enc_voc, sentence):
    return [enc_voc[word] for word in sentence] + [enc_voc["EOS"]]

def evaluate(searcher, enc_voc, dec_voc, sentence, max_length):
    ### 转换输入批次的编码内容为token_index
    # words -> indices
    indexes_batch = [indexesFromSentence(enc_voc, sentence)]
    # 创建长度张量
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 调换批次的大小以符合模型的预期
    input_batch = torch.LongTensor(indexes_batch)
    # 指定合适的设备
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # 通过searcher对编码内容进行解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = dec_voc.lookup_tokens(tokens.numpy())
    return decoded_words

def evaluateInput(searcher, enc_voc, dec_voc, max_length):
    input_sentence = ''
    while(1):
        try:
            # 获取用户输入
            input_sentence = input('> ')
            # 检查是否输入退出指令
            if input_sentence == 'q' or input_sentence == 'quit': break
            with torch.no_grad():
                # 调用模型生成解码内容
                output_words = evaluate(searcher, enc_voc, dec_voc, input_sentence, max_length)
            # 格式化并输出解码文本
            outputs = []
            for x in output_words:
                if x == 'EOS': break
                outputs.append(x)
            print('生成的内容:', ''.join(outputs))

        except KeyError:
            print("Error: 遇到未知token")

if __name__ == '__main__':
    # 设置加载指定epoch的存盘文件; 如果设置为None意味着从零开始
    
    # loadFilename = None
    
    checkpoint_epoch = 3
    save_dir = os.path.join("data", "save")
    corpus_name = "numeric convert corpus"
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, rnn_hidden_size),
                            'epoch_{}_{}.tar'.format(checkpoint_epoch, 'checkpoint'))


    if loadFilename:
        # 加载模型的参数
        # checkpoint = torch.load(loadFilename)
        # 还可以指定参数是否在加载在cpu环境上
        checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        encoder_embedding_sd = checkpoint['enc_embedding']
        decoder_embedding_sd = checkpoint['dec_embedding']
        enc_voc = checkpoint['enc_voc_dict']
        dec_voc = checkpoint['dec_voc_dict']
        dec_max_length = checkpoint['dec_max_length']
        
    print('构建编码器和解码器...')
    # 初始化embeddings
    encoder_embedding = nn.Embedding(len(enc_voc['vocab']), emb_hidden_size, padding_idx=0)
    decoder_embedding = nn.Embedding(len(dec_voc['vocab']), emb_hidden_size * 2, padding_idx=0)
    if loadFilename:
        encoder_embedding.load_state_dict(encoder_embedding_sd)
        decoder_embedding.load_state_dict(decoder_embedding_sd)
    # 初始化encoder和decoder模型
    encoder = EncoderRNN(rnn_hidden_size, encoder_embedding, encoder_n_layers, dropout)
    decoder = DecoderRNN(rnn_hidden_size * 2, decoder_embedding, len(dec_voc['vocab']), decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # 使用的设备
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('模型构建成功！')

    searcher = Inference(encoder, decoder)
    evaluateInput(searcher, enc_voc['vocab'], dec_voc['vocab'], dec_max_length)