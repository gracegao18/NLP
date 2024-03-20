# 自定义数据集
import torch
from torch.utils import data
import os
from preprocess import build_vocab, build_tag_dict, read_corpus

class NerDataSet(data.Dataset):
    """
    NER加载语料用Dataset
    """
    def __init__(self, tokens, tags, vocab, tag_to_ix):
        # 初始化数据集 通过内置属性来保存features和labels
        assert len(tokens) == len(tags), 'token和tags需要长度一致'
        self.tokens = [vocab(token) for token in tokens]
        self.tags = [[tag_to_ix[g] for g in tag] for tag in tags]

    def __getitem__(self, index):
        # 通过索引读取数据集中的index位置的元素
        token, tag = self.tokens[index], self.tags[index]
        # 数据必要的转换
        # 返回数据对 features,labels
        return token, tag
        
    def __len__(self):
        # 数据集的总大小
        return len(self.tokens)

if __name__ == '__main__':
    
    train_file = os.path.abspath('./chinese_ner/msra_bio/msra_train_bio')
    tag_file = os.path.abspath('./chinese_ner/msra_bio/tags.txt')
    # train_file = os.path.join(os.path.dirname(__file__), 'msra_bio/msra_train_bio')
    # tag_file = os.path.join(os.path.dirname(__file__), 'msra_bio/tags.txt')
    
    # 创建训练用语料
    train_tokens, train_tags = read_corpus(train_file)
    # 构建词汇表和tag字典
    vocab = build_vocab(train_tokens)
    tag_to_ix = build_tag_dict(tag_file)
    # 测试自定义Dataset
    ds = NerDataSet(train_tokens, train_tags, vocab, tag_to_ix)
    print(len(ds))