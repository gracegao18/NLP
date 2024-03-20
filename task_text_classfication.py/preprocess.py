import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer, re
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import SubsetRandomSampler, Subset

from torchtext.datasets import SogouNews

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_text_pipeline(vocab, tokenizer):
    # 把word转为word_index
    return lambda x: vocab(tokenizer(x))

def build_label_pipeline():
    # 把label自然索引转换为以0起始的索引
    return lambda x: int(x) - 1

def build_vocab(dataset):
    # 获取一个分词器(默认按空格拆分词汇)
    tokenizer = get_tokenizer(None)
    # 使用分词器对每个记录进行拆分
    tokens = (tokenizer(text) for _,text in dataset)
    # 构建词汇表
    vocab = build_vocab_from_iterator(tokens, specials=["<pad>","<unk>"])
    # 设置默认未知词汇的默认索引
    vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab

def build_data_loader(dataset, vocab, tokenizer, batch_size = None, shuffle = False):
    # pipeline
    text_pipeline = build_text_pipeline(vocab, tokenizer)
    label_pipeline = build_label_pipeline()

    def collate_batch(batch):
        label_list, text_list = [],[]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True)
        return label_list.to(device), text_list.to(device)

    # 构建数据加载器
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

if __name__ == '__main__':
    # 加载语料
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    train = SogouNews(root=data_path, split=('train'))
    # 构建词汇表和分词器
    tokenizer, vocab = build_vocab(train)
    
    # 重新加载
    train,test = SogouNews(root=data_path, split=('train','test'))
    # 构建data_loader
    train_dl = build_data_loader(train, vocab, tokenizer,  batch_size=32)
    test_dl = build_data_loader(test, vocab, tokenizer, batch_size=32)

    for label, text in test_dl:
        print(label.shape)
        print(text.shape)

