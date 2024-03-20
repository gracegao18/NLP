from turtle import forward
import torch
from torchtext.datasets import SogouNews
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch import batch_norm, nn  
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), type=torch.int64)
        text_list.append(processed_text)
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

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

class SomeModel(nn.Model):
    def __init__(self, vocab_size, emb_hidden_size, rnn_hidden_size, num_layers, num_class):
        super(SomeModel, self).__init__()
        # Embedding模型
        self.embedding = nn.Embedding(vocab_size, emb_hidden_size, padding_idx=0)
        #LSTM模型
        self.rnn = nn.LSTM(
            input_size = emb_hidden_size,
            hidden_size = rnn_hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )
        self.out = nn.Linear(rnn_hidden_size, num_class)
        
    def forward(self, x):
        out = self.embedding(x)
        r_out, (c_n, h_n) = self.rnn(out)
        out = self.out(r_out[:, -1, :])
        return out


# # 把文本语料集中的每个样本，统一转换为token_index的集合。
# for _label, _text in train:
#     tockens_index = vocab(tokenizer(_text))
#     # 查看转化后的list长度和list中的前10哥token_index
#     print('token_index length: %d \ntoken_index head: %r' %(len(tockens_index), tockens_index[:10]))

# 把token_index映射到Embedding -> 解决第二个问题
# emb = Embedding(vocab_size, hidden_size, padding_idx=0)


if __name__ == '__main__':
    # 加载数据集
    train = SogouNews(root='data', split='train')
    train.current_pos = 0
    #通过数据集创建分词器和词典对象
    tokenizer, vocab = build_vocab(train)
    
    # 把word转为word_index
    text_pipeline = lambda x: vocab(tokenizer(x))
    # 把label自然索引转换为以0起始的索引
    label_pipeline = lambda x: int(x)-1
    
    '''
    —— sogou语料库：拼音+注音 没有汉字，不太算完整的中文语料

    —— `root` - 数据集所在的目录(data代表读取代码当前目录下的data子目录)
    `split` - 获取train或test数据子集，可以是字符串'train'或'test'或包含这两个字符串的list
    
    —— 关于Token：在NLP的语料处理分词环节，因语言不同，往往也会产生出不同的结果。
    以东亚语言为例(中、日、韩)：我们既可以把文章拆分为“词”，又可以把文章拆分为“字”。
    这种结果导致我们在描述分词时往往会让人产生歧义，token可以解决这个概念(或者说描述)问题，
    不论我们拆分的是什么，每个被拆分出的内容，统一以token指代。
    词汇表中保存的是不重复的token，token_index也就是给每个token分配的唯一索引值。

    —— 关于OOV（Out Of Value)：OOV是指模型在测试或预测中遇到的新词。
    这些新词在模型已构建的词汇表中并不存在。因此，会导致模型方法无法识别并处理这些OOV新词。
    可以给未来的新词设置1个或多个特殊符号，这些token会插入在词表的顶端（索引为0的位置）。
    在构建词表时，使用"<unk>" 来表示unknown。之后再通过set_default_index(vocab["<unk>"]) 把词表的默认索引设置为"<unk>" 
    这意味着：当遇到词表中不存在的token时，默认使用"<unk>" 作为该token的索引。
    '''
    train, test = SogouNews(root='data', split=('train', 'test'))
    
    '''
    这些token_index不可以直接导入模型进行训练么，因为样本的特征信息太少了。
    可以通过词频矩阵转TF/IDF矩阵，但不是很有效。
    尤其是与时间序列相关的RNN模型，TF/IDF不能提供token之间的前后关联信息。
    NLP最常用的做法是：每个token映射一个对应的向量。把词汇表中所有token的向量组成矩阵，就称之为词嵌入(词向量矩阵) Embedding。
    下面要解决的问题有两个：
    1. 要解决不同长度文本的对齐。这样才可以在指定batch_size后，批量的把数据导入模型进行训练。
    2. 把token_index映射到Embedding。
    '''
    # 定制Dataloader，每batch_size的样本数据会经过collate_batch方法处理后再传给模型 -> 解决第一个问题
    dataloader = DataLoader(train, batch_size=8, shuffle=False, collate_fn=collate_batch)
    model = SomeModel(len(vocab), 128, 128, 1, 5)
    model.to(device)
    # # 创建分词器对象（默认安空格拆分词汇）
    # tokenizer = get_tokenizer(None)
    # # 使用分词器对每个记录进行拆分
    # tokens = (tokenizer(text) for _, text in train)
    # # 构建词汇表
    # vocab = build_vocab_from_iterator(tokens, specials=["<pad>", "<unk>"])
    # # 设置默认未知词汇的默认索引
    # vocab.set_default_index(vocab["<unk>"])
    
    for label, text in dataloader:
        print(label)
        print(text)
        # text.to(device)
        # out = model(text)
        # print(out)
        # break        
    
    