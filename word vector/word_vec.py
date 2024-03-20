# from gensim import models
# from gensim.models import Word2Vec, word2vec

# # 导入语料
# sentences = word2vec.Text8Corpus('msr_training.utf8')

# model = Word2Vec(sentences=sentences, vector_size=200)
# model.wv.save_word2vec_format('word2vec.model', binary=True)

import os
from gensim.models import Word2Vec, word2vec
from gensim.models.keyedvectors import KeyedVectors

# 语料文件
corpus_file = os.path.join(os.path.dirname(__file__), 'zhihu.txt')
# 模型二进制存盘文件
save_bin_file = os.path.join(os.path.dirname(__file__), 'embeddings', 'word2vec.model')
# 模型文本格式存盘文件
save_txt_file = os.path.join(os.path.dirname(__file__), 'embeddings', 'word2vec.txt')

def train_word2vec():
    # 如果没有整理好文本（分词，去掉'\n or \r or ...'）则删掉：word2vec.Text8Corpus(corpus_file)
    sentences = word2vec.Text8Corpus(corpus_file) # 8 meaning ‘by’
    model = Word2Vec(sentences=sentences, vector_size=200)
    model.wv.save_word2vec_format(save_bin_file, binary=True)
    model.wv.save_word2vec_format(save_txt_file)

def load_vectors():
    # 加载存盘的词向量模型
    word_vectors = KeyedVectors.load_word2vec_format(save_bin_file, binary=True)
    # word_vectors = KeyedVectors.load_word2vec_format(save_txt_file)
    return word_vectors

if __name__ == '__main__':
    # train_word2vec()

    wv = load_vectors()

    # 词汇数量
    print('词典大小:',len(wv.key_to_index))
    # 词汇相似度
    print(wv.similarity('新闻','番茄'))
    # 获取最近似的词汇
    print(wv.most_similar("机遇", topn=10))