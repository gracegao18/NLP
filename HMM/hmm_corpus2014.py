import os
import numpy as np
import tarfile
import pickle
import jieba

from hmm_train_p import get_tag, mle_train, HMM, viterbi, build_viterbi_path, state_path, GB2312_K

# 数据文件所在目录
base_dir = os.path.join(os.path.dirname(__file__),'nlp_data')
# 模型文件所在目录
model_dir = os.path.join(os.path.dirname(__file__),'models')
# gz压缩格式语料文件
corpus_file = '2014_corpus.txt'

state_label2id = {'B':0, 'M':1, 'E':2, 'S':3 }
state_id2label = { v:k for k,v in state_label2id.items()}

def load_corpus():
    """加载并处理分词语料"""
    # 打开文件 获取所有句子
    # lines = []
    # tar = tarfile.open(os.path.join(base_dir, corpus_file), 'r:gz', encoding='utf-8')
    # for tarinfo in tar:
    #     # 读取文件内容
    #     if tarinfo.isreg():
    #         f = tar.extractfile(tarinfo)
    #         ctx = f.read()
    #         lines.append(ctx.decode("utf-8"))
    
    with open(os.path.join(base_dir, '2014_corpus.txt'),'r', encoding='utf-8') as fh:
        lines = fh.read().split('\n')

    print("语料数量:", len(lines))
    temps = []
    for l in lines:
        words = l.split()
        words = [word[:word.find('/')].replace('[','') for word in words]
        temps.append(' '.join(words))
    lines = temps
    return lines

def train_model(lines):
    """训练HMM模型"""
    pi, A, B = mle_train(lines)

    # 构建 HMM 模型，进行预测
    np_pi = np.zeros(len(pi), dtype=float)
    np_mA = np.zeros((len(pi),len(pi)),dtype=float)

    for k, k_id in state_label2id.items():
        # 把pi转换为向量：
        np_pi[k_id] = pi[k]
        for k2, k2_id in state_label2id.items():
            # 把A转换为4x4矩阵: 
            np_mA[k_id][k2_id] = A[k][k2]

    idmB = dict((state_label2id[key], value) for (key, value) in B.items())
    hmm_seg_model = HMM(np_pi, np_mA, idmB)
    return hmm_seg_model

def save_model(hmm_model, pickle_file):
    """保存HMM模型"""
    if isinstance(hmm_model, HMM):
        with open(pickle_file,'wb') as f:
            pickle.dump(hmm_model,f)
            print('模型保存成功!')
    else:
        raise Exception('模型保存失败')

def load_model(pickle_file):
    """加载HMM模型"""
    try:
        with open(pickle_file,'rb') as f:
            hmm_model = pickle.load(f)
        return hmm_model
    except Exception as e:
        print('模型加载失败!', e)


if __name__ == '__main__':

    # 打开文件 加载并处理语料
    lines = load_corpus()
    # 训练HMM模型
    hmm_model = train_model(lines)
    # 保存模型
    save_model(hmm_model, os.path.join(model_dir,'hmm.pkl'))

    # 加载模型
    hmm_model = load_model(os.path.join(model_dir,'hmm.pkl'))

    obs = [
        "中国人越来越不愿意生孩子，已经不仅影响未来几代人，而是一个现实的危机，这说明养儿防老的中国传统养老方式，基本难以维系了。养老问题的严重程度，已经到了不可忽视的地步。",
        "告别了阴雨连绵的春天，迎来了浩浩荡荡的夏天，每一阵风中，仿佛都洋溢着生命的热情，奔放而可爱。",
        "花了太久时间阐述兄弟为何反目成仇，冲突解释的力度不大但是可以接受；解释韩为什么复活那段，过程差点睡着（因为韩在解释的时候表情就像在撒谎），韩的死而复生居然仅仅因为无名氏的通天本领！“无名氏有很多办法，你知道的”，我知道个屁，无名氏那么dio咋还让人抓了。" \
        "不抠细节，剧情本来就没抱期待，拖节奏是因为要填之前没填的坑，每一部都换导演真的太乱了。。。所以宽容地期待着劲爆场面，赵喜娜的加入并没有带来惊喜，作为剧情的补充也是用的青年演员出演，他带来的仅仅是那张WWE的硬汉脸，就花拳绣腿了两分钟左右，也许是由于剧情设计他是个弱鸡，所以怎么看都没有强森给的冲劲大，感觉就是来打酱油的。" \
        "飙车戏一如既往的酣畅淋漓，尤其是配着激情四射的BGM，美女、跑车、引擎的轰鸣声是最能激发男性荷尔蒙的良药。" \
        "地球已经容不下特雷托家族了，所以这一部，咱上太空，很多bug整的牛顿的棺材板都要压不住了，以前几部也挺扯，但是没有这一部那么…天马行空，果然，贫穷限制了我的想象力，还是觉得温子仁那部完结了就岁月静好。"]

    cs = []
    for o in obs:
        V,prev = viterbi(hmm_model, o)
        score, path_iter = state_path(V, prev)

        jieba_words = list(jieba.cut(o))

        # print(score)o
        split_words = []
        word = ''
        for i, s in zip(path_iter, o):
            word += s
            if state_id2label[i] in ['E','S']:
                split_words.append(word)
                word = ''
        print('*'*50,'HMM分词','*'*50)
        print(split_words)
        print('*'*50,'jieba分词','*'*50)
        print(jieba_words)

        correct = list(set(split_words).union(set(jieba_words)))
        cs.append(len(correct) / len(jieba_words))
        
    print('准确率:%.2f%%'%(np.mean(cs) * 100))