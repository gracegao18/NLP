from sys import path
import jieba
import os
from  sklearn.cluster import KMeans
import numpy as np

from  gensim.models import Word2Vec
from gensim import corpora

if __name__ == '__main__':
    artical_words = []
    corpus_file = os.path.join(os.path.dirname(__file__), 'lee_background_zh.txt')
    stopwords_file = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    
    embedding_file = os.path.join(os.path.dirname(__file__), 'embedding', 'word2vec.txt')
    
    # 停用词列表
    stop_words = [line.strip() for line in open(stopwords_file, encoding='utf-8').readlines()]
    
    for line in open(corpus_file, encoding='utf-8'):
        words = jieba.lcut(line.strip())
        temp_list = filter(lambda c: c not in stop_words, words)
        artical_words.append(list(temp_list))
    # print(artical_words)
    
    # 制作词汇字典
    dictionary = corpora.Dictionary(artical_words)
    id2token = {v: k for k, v in dictionary.token2id.items()}
    
    # 词汇转换为索引并计数（bag of word）
    all_words = []
    for words in artical_words:
        all_words.extend(words)
        
    # 计算词频: doc2bow()
    index_frequen = dictionary.doc2bow(all_words)
    word_frequen = [(id2token[t[0]], t[1]) for t in index_frequen]
    
    # 词频排序
    sorted_freq = sorted(word_frequen, key=lambda x:x[1], reverse=True)
    print(sorted_freq[:30])