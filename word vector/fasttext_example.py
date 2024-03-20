import os
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

# 训练语料文件
corpus_file = datapath('lee_background.cor')
# 模型存盘文件
save_model_file = os.path.join(os.path.dirname(__file__),'embeddings', 'fasttext.model')
# 模型word2vec格式文本存盘文件
save_txt_file = os.path.join(os.path.dirname(__file__),'embeddings', 'fasttext.txt')

def train_fasttext():
    model = FastText(vector_size=100)
    # 构建词汇表
    model.build_vocab(corpus_file=corpus_file)
    # 训练模型
    model.train(
        corpus_file=corpus_file, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words,
    )
    # 保存模型
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_txt_file)

def load_fasttext_model():
    loaded_model = FastText.load(save_model_file)
    return loaded_model

if __name__ == '__main__':
    # train_fasttext()
    model = load_fasttext_model()

    wv = model.wv
    print(wv['night'])

    print(wv.similarity("night", "nights"))

    print(wv.most_similar("nights"))

    print(wv.doesnt_match("breakfast cereal dinner lunch".split()))