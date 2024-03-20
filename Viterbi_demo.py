import numpy as np

from HMM_概率计算问题 import B

def decoder(model, observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_states = np.shape(model["pi"])[0]
    
    # 记录从 t-1～t时刻的状态 i
    # 最可能从哪个状态转移来，假设状态为：j
    psi = np.zeros([N_samples, N_states])
    
    # 从t-1～t时刻的状态，状态j～状态i的最大转移概率
    delta = np.zeros([N_samples, N_states])
    
    # 初始化
    delta[0] = model["pi"]*model["B"](model, o[0])
    psi[0] = 0
    
    
    # 递推填充 delta & psi
    for t in range(1, N_samples):
        for i in range(N_states):
            states_prev2current = delta[t-1]*model["A"][:, i]
            delta[t][i] = np.max(states_prev2current)
            psi[t][i] = np.argmax(states_prev2current)
        delta[t] = delta[t]*model["B"](model, o[t])
        
    # 反向回溯寻找最佳路径：计算最大时刻
    path = np.zeros(N_samples)
    path[-1] = np.argmax(delta[-1])
    prob_max = np.max(delta[-1])
    
    # 回溯
    for t in range(N_samples-2, -1, -1):
        path[t] = psi[t+1][int(path[t+1])]
        
    return prob_max, path

# 观测样本o属于各个隐藏状态s的概率
def prob_O2S(model,o):
    M_O2S = model["M_O2S"]
    print(M_O2S[:,int(o)])
    return M_O2S[:,int(o)]


'''设计HMM模型'''
# 构建一个已知参数的HMM模型
model_hmm = dict()
# 状态的初始分布
model_hmm["pi"] = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
# 各个状态之间的转移关系
# row：当前状态，col：下一状态
model_hmm["A"] = np.array([
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4],
])
# 观测样本与各个状态之间的概率映射关系矩阵
M_O2S = np.zeros([3, 8])
M_O2S[0, :6] = 1/6.0
M_O2S[1, :4] = 1/4.0
M_O2S[2, :8] = 1/8.0
model_hmm["M_O2S"] = M_O2S
# 计算观测样本o 属于状态S的概率的函数
model_hmm["B"] = prob_O2S


'''Viterbi算法使用'''
datas = np.array([0, 1, 0, 1])
_, path = decoder(model_hmm, datas)
print(path)