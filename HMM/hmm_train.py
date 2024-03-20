import os
import numpy as np

# 数据文件所在目录
base_dir = os.path.join(os.path.dirname(__file__),'nlp_data')

# GB2312字符集 字符个数6763
GB2312_K = 6763 

class HMM:
    """
    A : 状态转移概率矩阵
    B : 观测概率分布矩阵
    pi : 初始状态概率向量
    """
    def __init__(self, pi, A, B):
        self.A = A
        self.B = B
        self.pi = pi

def get_tag(token):
    token_len = len(token)
    if token_len == 1:
        return 'S'
    m_count = token_len - 2
    return ''.join(['B', 'M'*m_count, 'E'])

# 测试BMES标记
# tag = get_tag('最大似然')
# print(tag)


with open(os.path.join(base_dir, '2014_corpus.txt'),'r', encoding='utf-8') as fh:
    lines = fh.read().split('\n')

def mle_train(samples):
    """构造矩阵"""
    mA = dict() # 状态转移矩阵
    mB = dict() # 观测概率矩阵
    vecPi = dict() # 初始状态概率
    state_list = list('BMES')  
    # B-Begin 词汇开始标记   '天'  '速'
    # M-Mid  词汇中间标记    '安'  
    # E-End  词汇结束标记    '门'  '度'
    # S-Single 单个字       '是'

    # 初始化
    for state in state_list:
        # 初始状态概率向量
        vecPi[state] = 0.0
        mA[state] = {}
        mB[state] = {}
        # 状态转移概率矩阵 A
        for state1 in state_list:
            mA[state][state1] = 0.0

    # 遍历语料中每行的句子
    for line in samples:
        # 拆分句子中的词汇
        tokens = line.split(' ')
        # filter函数, 过滤掉不符合条件的元素(空字符串)，返回由符合条件元素组成的新列表
        tokens = list(filter(lambda x: x != '', tokens))
        tags = []

        # 遍历句子中的token(词汇)列表(前后词汇是合理的顺序)
        for t in range(0, len(tokens)):
            token= tokens[t]
            # 获取每个词的tag(BMES)标记
            tag = get_tag(token)
            tags.append(tag)

            if t == 0:
                # 初始状态 pi (两种可能状态: B或S)
                vecPi[tag[0]] += 1
            
            # 处理mB, 记录不同状态下的观测值:
            #  { 'B':{'字':1,'符':1,...,'序':1,'列':1},
            #    'M':{'其':1,'它':1,...,'字':1,'符':1},
            #    'E':{'内':1,'容':1,...,'不':1,'同':1},
            #    'S':{'根':1,'据':1,...,'文':1,'本':1}}
            for char, char_tag in zip(token, tag):
                mB[char_tag][char] = mB[char_tag].get(char,0) + 1

        # 处理 mA, 错位对齐
        for prev_tag, tag in zip(''.join(['_'] + tags), ''.join(tags + ['_'])):
            if prev_tag == '_' or tag == '_':
                continue
            mA[prev_tag][tag] += 1
        
    def probs(Pi, mA, mB):
        """根据出现的次数计算不同矩阵的概率值"""

        # Pi向量的初始状态概率
        totalPi = 0
        # Pi总数 = Pi中 B,M,E,S 标记的总数
        for v in Pi.values():
            totalPi += v
        # {'B': B标记的总数/Pi总数, 'M':B标记的总数/Pi总数,
        #  'E': E标记的总数/Pi总数, 'S':S标记的总数/Pi总数  }
        Pi = dict((key, value / totalPi) for (key, value) in Pi.items())

        # A矩阵中 B,M,E,S 每个标记的状态转移概率值
        for state in state_list:
            totalA = 0
            for v in mA[state].values():
                totalA += v
            # B:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到S的概率}
            # M:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到S的概率}
            # E:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到S的概率}
            # S:{ 转到B的概率, 转到M的概率, 转到E的概率, 转到S的概率}
            mA[state] = dict((key, value/totalA) for (key, value) in mA[state].items())

        # B矩阵中 B,M,E,S 每个标记的不同观测值(字符)的概率
        for state in state_list:
            totalB = 0
            for v in mB[state].values():
                totalB += v
            # 拉普拉斯平滑：为避免未出现的观测值为0而导致概率为0的情况，给分子和分母都+1
            # mB[state] = dict((key,(value+1)/(totalB + GB2312_K)) for (key, value) in mB[state].items())
            mB[state] = dict((key,(value+1)/(GB2312_K+1)) for (key, value) in mB[state].items())
        return Pi, mA, mB

    return probs(vecPi, mA, mB)     

def viterbi(hmm, obs_seq):

    # 每个观测值的默认概率
    smooth_prob = 1.0 / GB2312_K

    # 4种状态(BMES)
    N = hmm.pi.shape[0]
    # 观测序列的长度(计算的时间步)
    T = len(obs_seq)
    # 前向递推计算的结果集,初始为0
    # 记录了t时刻观测序列概率的最大值
    V = np.zeros((T, N))
    # 每一时刻的 B,M,E,S 状态的
    prev = np.zeros((T, N), dtype=int)
    
    def getBprob(c):
        # 返回 B,M,E,S 所有状态下，观测值c的概率
        return [hmm.B[n].get(c, smooth_prob) for n in range(0, N)]

    # 计算t0步结果
    V[0] = hmm.pi * np.asarray(getBprob(obs_seq[0]))

    # 计算剩下的时间步
    for t in range(1, T):
        # 每种状态结果的概率
        for n in range(N):
            # seq_probs = (V[:,t-1] * hmm.A[:,n]) * hmm.B[n].get(obs_seq[t], smooth_prob)
            seq_probs = (V[t-1] * hmm.A[:,n]) 
            V[t][n] = np.max(seq_probs)
            prev[t][n] = np.argmax(seq_probs)
            
        V[t] = V[t] * hmm.B.get(obs_seq[t], smooth_prob)
        
    # 反向回溯寻找最佳路径
    path = np.zeros(T)
    path[-1] = np.argmax(V[-1])
    prob_max = np.max(V[-1])
    
    for t in range(T-2,-1,-1):
        path[t] = prev[t+1][int(path[t+1])]
        
    return prob_max,path


# def build_viterbi_path(prev, last_state):
#     T = len(prev)
#     yield (last_state)
#     for i in range(T-2, -1, -1):
#         yield (prev[i, last_state])
#         last_state = prev[i+1][int(T[i+1])]

# def state_path(V, obs):
    # last_state = np.argmax(V[-1])
    # path = list(build_viterbi_path(prev, last_state))
    # return V[last_state, -1], reversed(path)


if __name__=='__main__':
    
    # 数据文件所在目录
    base_dir = os.path.join(os.path.dirname(__file__),'nlp_data')

    # 打开文件 获取所有行记录
    with open(os.path.join(base_dir, '2014_corpus.txt'),'r', encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    
    pi, A, B = mle_train(lines)

    # print(pi)
    # print(A)
    # print(B)

    # 构建 HMM 模型，进行预测
    state_label2id = {'B':0, 'M':1, 'E':2, 'S':3 }
    state_id2label = { v:k for k,v in state_label2id.items()}

    np_pi = np.zeros(len(pi), dtype=float)
    np_mA = np.zeros((len(pi),len(pi)),dtype=float)

    for k, k_id in state_label2id.items():
        # 把pi转换为向量：
        # [0.6345649207515482, 0,0, 0.0, 0.36543507924845176]
        np_pi[k_id] = pi[k]
        for k2, k2_id in state_label2id.items():
            # 把A转换为4x4矩阵: 
            # [[0.0, 0.14677577551236617, 0.8532242244876338, 0.0],
            #  [0.0, 3456713007046277, 0.6543286992953723, 0.0],
            #  [0.48537965431249463, 0.0, 0.0, 0.5146203456875054],
            #  [0.5710373878922667, 0.0, 0.0, 0.42896261210773334]]
            np_mA[k_id][k2_id] = A[k][k2]

    # 把 B中的key:'B','M','E','S' 换成 0,1,2,3
    idmB = dict((state_label2id[key], value) for (key, value) in B.items())
    hmm_seg_model = HMM(np_pi, np_mA, idmB)

    obs = "中国人越来越不愿意生孩子，已经不仅影响未来几代人，而是一个现实的危机，这说明养儿防老的中国传统养老方式，基本难以维系了。养老问题的严重程度，已经到了不可忽视的地步。"
    
    V,prev = viterbi(hmm_seg_model, obs)
    # score, path_iter = state_path(V, obs)
    # print(score)
    # print(V)
    
    # for i, s in zip(path_iter, obs):
    #     sp = ' ' if 2 <= i <= 3 else ''
    #     print(f'{s}{sp}',end='')
        
    for i, s in zip(prev, obs):
        sp = ' ' if 2 <= i <= 3 else ''
        print(f'{s}{sp}',end='')
        print("%s%s"%(s, state_id2label[i]), end='')