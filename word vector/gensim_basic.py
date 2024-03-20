import os
import jieba
import logging as log
from gensim import corpora

log.basicConfig(level=log.DEBUG)

corpus_file = os.path.join(os.path.dirname(__file__), '世界这么大还是遇见你.txt')

def read_text(corpus_file):
    """读取语料并进行分词"""
    text_words = []
    for line in open(corpus_file, encoding='utf-8'):
        line = line.strip()
        if line == "":
            continue
        text_words.append(jieba.lcut(line))
    return text_words

if __name__ == '__main__':
    text = read_text(corpus_file)
    log.debug(text)
    
    # 制作词汇字典
    dictionary = corpora.Dictionary(text)
    log.debug(dictionary.token2id)
    
    # 词汇转换为索引并计数（bag of word），一句话为一次计数
    log.debug(dictionary.doc2bow('多少 次 疯狂 的 旅程'.split()))
    
    # 词汇转换为数学表示（矢量）
    bowcorpus = [dictionary.doc2bow(l) for l in text]
    log.debug(bowcorpus)
    