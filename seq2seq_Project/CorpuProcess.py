import os
import torch
import numpy as np
import logging as log
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

log.basicConfig(level=log.DEBUG)

corpus_file = os.path.join(os.path.dirname(__file__),'dataset.txt')

def read_corpus(corpus_file):
    """读取语料文件"""
    with open(corpus_file,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')

    encode_datas,decode_datas = [],[]
    for line in lines:
        if not line.strip():
            continue  # 跳过空行
        try:
            input_text, target_text = line.split('\t')
            encode_datas.append(input_text)
            decode_datas.append(target_text)
        except ValueError:
            log.error('Error line: %s'%line)
            input_text=''
            target_text=''

    return encode_datas, decode_datas

class Seq2SeqDataset(Dataset):

    def __init__(self, encode_datas, decode_datas):
        self.encode_datas = encode_datas
        self.decode_datas = decode_datas
        self.encode_vocab = self.build_vocab(encode_datas,fill_mask = ["PAD", "EOS"])
        self.decode_vocab = self.build_vocab(decode_datas,fill_mask = ["PAD", "SOS", "EOS"])

    def __getitem__(self,index):
        enc = list(self.encode_datas[index]) + ["EOS"]
        dec = ["SOS"] + list(self.decode_datas[index]) + ["EOS"]
        e = self.encode_vocab(enc)
        d = self.decode_vocab(dec)
        return e,d
    
    def __len__(self):
        return len(self.encode_datas)

    def build_vocab(self, datas, fill_mask):
        """构建词汇表"""
        vocab = build_vocab_from_iterator(datas, specials=fill_mask)
        return vocab

def build_dataloader(dataset, batch_size = None, shuffle = False):
    def collate_batch(batch):
        encode_list, decode_list, enc_len_list = [],[],[]
        for encode,decode in batch:
            processed_enc = torch.tensor(encode, dtype=torch.int64)
            processed_dec = torch.tensor(decode, dtype=torch.int64)
            encode_list.append(processed_enc)
            enc_len_list.append(len(encode))
            decode_list.append(processed_dec)
        encode_list = pad_sequence(encode_list, batch_first=True)
        decode_list = pad_sequence(decode_list, batch_first=True)
        enc_len_list = torch.tensor(enc_len_list, dtype = torch.int64)
        return encode_list, enc_len_list, decode_list, decode_list > 0

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

if __name__ == '__main__':
    encode_datas,decode_datas = read_corpus(corpus_file)

    dataset = Seq2SeqDataset(encode_datas, decode_datas)
    dataloader = build_dataloader(dataset, batch_size=4, shuffle=True)
    for enc,enc_len,dec,dec_mask in dataloader:
        print(enc.shape)
        print(dec.shape)
        print(enc)
        print(dec)
        print(enc_len)
        print(dec_mask)
        break
    

    