import numpy as np

class HMM:
    """
    Order 1 Hidden Markov Model
    ATtributes
    ----------
    A : numpy.ndarray
        状态转移概率矩阵
    B : numpy.ndarray
        观测概率分布矩阵 (N, number of output types)
    pi : numpy.ndarray
        初始状态概率向量

    Common Variables
    ----------------
    obs_seq : list of int
        list of obervations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, pi, A, B):
        self.A = A
        self.B = B
        self.pi = pi


# 对应状态集合 S
states = ('健康','发烧') 

# 对应观测集合 O
observations = ('正常', '发冷', '头晕')    

# 初始状态概率向量 π
start_probability = {'健康':0.6, '发烧':0.4}

# 状态转移矩阵 A
transition_probability ={
    '健康': {'健康':0.7, '发烧':0.3},
    '发烧': {'健康':0.4, '发烧':0.6}
}

# 观测概率矩阵 B
emission_probability ={
    '健康': {'正常':0.5, '发冷':0.4, '头晕':0.1},
    '发烧': {'正常':0.1, '发冷':0.3, '头晕':0.6}
}

# 对状态进行编号
def generate_index_map(labels):
    id2label = {} # 2 == to
    label2id = {}
    
    for i,l in enumerate(labels):
        id2label[i] = l
        label2id[l] = i
    return id2label, label2id

# 对状态集合 S 进行编号
states_id2label, states_label2id = generate_index_map(states)

# 对观测集合 O 进行编号
observations_id2label, observations_label2id = generate_index_map(observations)

def convert_map_to_vector(map_, label2id):
    """将概率向量丛dict转换成一维array"""
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v

def convert_map_to_matrix(map_, label2id1, label2id2):
    """将概率转移矩阵从dict转换成矩阵"""
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for row in map_:
        for col in map_[row]:
            m[label2id1[row]][label2id2[col]] = map_[row][col]
    return m

A = convert_map_to_matrix(
    transition_probability,
    states_label2id, states_label2id)

B = convert_map_to_matrix(
    emission_probability,
    states_label2id, observations_label2id)

observations_index = [observations_label2id[o] for o in observations]

pi = convert_map_to_vector(
    start_probability,
    states_label2id)

print("\n初始状态 π:", pi, "\n'健康':0.6, '发烧':0.4")
print("\n状态转移概率矩阵 A:",A, "\n'健康': {'健康':0.7, '发烧':0.3},\n'发烧': {'健康':0.4, '发烧':0.6}")
print("\n观测概率分布矩阵 B:", B, "\n'健康': {'正常':0.5, '发冷':0.4, '头晕':0.1},\n'发烧': {'正常':0.1, '发冷':0.3, '头晕':0.6}")

# 构建模型
model = HMM(pi, A, B)

# 随机观测序列
o_seq = ['正常', '正常', '发冷', '发冷', '头晕',
         '发冷', '头晕', '头晕', '头晕', '正常']

"""
Q1： 观测序列与模型的匹配程度(概率计算问题)
Q2： 通过观测序列对病人状态进行判断 (解码问题)
"""

o_seq_ids = [observations_label2id[index] for index in o_seq]

print(o_seq_ids)

def hmm_forward(hmm, obs_seq):
    """
    A = NxN
    B = NxM
    """
    # 所有的状态(可能的结果)
    N = hmm.A.shape[0]
    # 序列的长度(计算的时间步)
    T = len(obs_seq)
    # 前向递推的结果集，初始为0
    F = np.zeros((N, T))
    # 计算第t0步结果: π * B[o1]
    # [0.6,0.4] * [0.5,0.1]
    F[:,0] = hmm.pi * hmm.B[:,obs_seq[0]]

    # 计算t1,t2,t3....tn步的结果
    for t in range(1, T):
        # 每一步都要计算的状态值
        for n in range(N):
            F[n,t] = np.dot(F[:,t-1], (hmm.A[:,n])) * hmm.B[n, obs_seq[t]]
    return F

f = hmm_forward(model, o_seq_ids)
print(f)
print(np.argmax(f[:, -1]))
