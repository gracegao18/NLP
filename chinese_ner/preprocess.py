# 语料预处理
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_corpus(corpus_path):
    """
    读取语料库中文本，并返回list样本
    :param corpus_path:语料文件完整路径
    :return: data
    """
    corpu_tokens, corpu_tags, tokens, tags = [],[],[],[]
    # 逐行读取语料文件
    for line in open(os.path.join(corpus_path), encoding='utf-8'):
        if line.strip() == '':
            corpu_tokens.append(tokens)
            corpu_tags.append(tags)
            tokens, tags = [],[]
            continue
        # 读取的行记录被分为: token和tag
        items = line.strip().split()
        if len(items) < 2: continue
        token,tag = items
        tokens.append(token)
        tags.append(tag)
    return corpu_tokens, corpu_tags

def build_vocab(tokens):
    """
    构建token字典表
    """
    # 构建token字典表
    vocab = build_vocab_from_iterator(tokens, specials=["<pad>","<unk>"])
    # 设置默认未知词汇的默认索引
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def build_tag_dict(tag_file):
    """
    通过tag文件构建tag字典表
    """
    lines = [line.strip() for line in open(tag_file)]
    tags = { tag:i for i,tag in enumerate(lines + ['START_TAG','STOP_TAG'])}
    return tags

def build_text_pipeline(vocab):
    # 把word转为word_index
    return lambda x: vocab(x)

def build_label_pipeline(tag_to_ix):
    # 把label自然索引转换为以0起始的索引
    return lambda x: [tag_to_ix[t] for t in x]

def build_data_loader(dataset, batch_size = None, shuffle = False):
    
    def collate_batch(batch):
        label_list, text_list = [],[]
        for (_text, _label) in batch:
            processed_label = torch.tensor(_label, dtype=torch.int64)
            label_list.append(processed_label)
            processed_text = torch.tensor(_text, dtype=torch.int64)
            text_list.append(processed_text)
        label_list = pad_sequence(label_list, batch_first=True)
        text_list = pad_sequence(text_list, batch_first=True)
        
        return text_list.to(device), label_list.to(device) 

    # 构建数据加载器
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

if __name__ == '__main__':
    
    train_file = os.path.abspath('./chinese_ner/msra_bio/msra_train_bio')
    tag_file = os.path.abspath('./chinese_ner/msra_bio/tags.txt')
    # train_file = os.path.join(os.path.dirname(__file__), 'msra_bio/msra_train_bio')
    # tag_file = os.path.join(os.path.dirname(__file__), 'msra_bio/tags.txt')
    
    tokens,tags = read_corpus(train_file)
    print('训练语料大小:',len(tokens))
    vocab = build_vocab(tokens)
    print('token字典大小:',len(vocab))
    tags = build_tag_dict(tag_file)
    print(tags)