import os
from gensim import models
from gensim.utils import chunkize
import jieba
from adjustText import adjust_text
from gensim.models import Word2Vec, keyedvectors
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary, dictionary
from gensim.models import LdaModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 语料文件
corpus_file = os.path.join(os.path.dirname(__file__), 'lee_background_zh.txt')
# 模型存盘文件
save_vec_file = os.path.join(os.path.dirname(__file__), 'lee_background_zh.bin')
# 停用词文件
stopwords_file = os.path.join(os.path.dirname(__file__), 'stopwords.txt')

# 停用词列表
stop_words = [line.strip() for line in open(stopwords_file, encoding='utf-8').readlines()]

def word2vec_generation(sentences):
    """
    生成词向量
    :param sentences: 分词后的list或带有空格的文本
    """
    model = Word2Vec(sentences=sentences, vector_size=200, window=5)
    return model

def load_word2vec(save_file):
    return Word2Vec.load(save_file)

def word_split(doc_file):
    """
    文档分词
    :param doc_file: 待分词文件    
    :return 分词后list
    """
    doc_words = []
    chapter_words = []
    for line in open(doc_file, encoding='utf-8'):
        if line.strip() == '':
            continue
        words = jieba.lcut(line.strip())
        temp_list = filter(lambda c: c not in stop_words, words)
        chapter_words.extend(list(temp_list))
        if line.strip() == '---------':
            doc_words.append(chapter_words)
            chapter_words = []
    doc_words.append(chapter_words)
    return doc_words
    
def extract_topics(words, topic_num=1, extract_words=35):
    """
    LDA算法提取文章主题
    :param words: 要提取的文章词汇集合
    :param topic_num: 提取的主题数量
    :param extract_words: 每个主题包含的关键词数量
    """
    dictionary = Dictionary(words)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    
    # 文档的词袋表示
    corpus = [dictionary.doc2bow(doc) for doc in words]
    
    # 设置LDA模型训练参数
    num_topics = topic_num
    chunkize = 2000
    passes = 20
    iteration = 400
    eval_every = None 
    
    temp = dictionary[0]    # This is only to "load" the dictionary
    id2word = dictionary.id2token
    
    model = LdaModel(
        corpus = corpus,
        id2word = id2word,
        chunksize = chunkize,
        alpha = 'auto',
        eta = 'auto',
        iterations = iteration,
        num_topics = num_topics,
        passes = passes,
        eval_every = eval_every
    )
    
    top_topics = model.top_topics(corpus, topn=extract_words)
    return top_topics

if __name__ == '__main__':
    # 分词：以章节划分
    words = word_split(corpus_file)
    # 生成词向量并保存
    model = word2vec_generation(words)
    model.wv.save_word2vec_format(save_vec_file, binary=True)
    
    """
    information stored in file is incomplete (the binary tree is missing), so while you can query for word similarity etc., you cannot continue training with a model loaded this way.
    using load_word2vec_format():  unicode_errors=...: default: 'strict',  
    if file may include word tokens truncated in the middle of a multibyte unicode character: use 'ignore'/ 'replace'.
    """
    word_vec = keyedvectors.load_word2vec_format(save_vec_file, binary=True, unicode_errors='ignore')
    
    topics = extract_topics(words)
    # topics[i][j]: i: num of topics, j: num of the item of the i's topic
    topic = topics[0][0]
    custom_vectors = {}
    # 遍历每个topic找关键词和词向量
    for item in topic:
        # 提取关键词
        keyword = item[1]
        custom_vectors[keyword] = word_vec[keyword]
        # 关键词最近似的20个词
        similars = word_vec.most_similar(keyword, topn=50)
        for sitem in similars:
            custom_vectors[keyword] = word_vec[sitem[0]] 
            
    keywords, vectors = [], []
    for k, v in custom_vectors.items():
        keywords.append(k)
        vectors.append(v)
        
    # 词向量维度降到2维：PCA()
    pca = PCA(n_components=2)
    coordinates_2d = pca.fit_transform(vectors)
    
    # 绘制散点图表达相关性
    plt.scatter(coordinates_2d[:, 0], coordinates_2d[:, 1])
    texts = [plt.text(coordinates_2d[i, 0], coordinates_2d[i, 1], keywords[i]) for i in range(len(keywords))]
    adjust_text(texts)
    plt.show()

