import numpy as np

def decoder(model,observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    # 纪录了从t-1到t时刻，状态i
    # 最有可能从哪个状态（假设为j）转移来的
    psi = np.zeros([N_samples,N_stats])
    
    # 从t-1到t时刻状态 状态j到状态i的最大转移概率
    delta = np.zeros([N_samples,N_stats])
    
    # 初始化
    delta[0] = model["pi"]*model["B"](model,o[0])             # 初始化
    psi[0] = 0
    
    # 递推填充delta与psi                                       # 递推
    for t in range(1,N_samples):
        for i in range(N_stats):
            states_prev2current = delta[t-1]*model["A"][:,i]
            delta[t][i] = np.max(states_prev2current)
            psi[t][i] = np.argmax(states_prev2current)
            
        delta[t] = delta[t]*model["B"](model,o[t])
        
    # 反向回溯寻找最佳路径                                      # 计算T时刻最大delta
    path = np.zeros(N_samples)
    path[-1] = np.argmax(delta[-1])
    prob_max = np.max(delta[-1])
    
    for t in range(N_samples-2,-1,-1):                        #利用psi进行回溯
        path[t] = psi[t+1][int(path[t+1])]
        
    return prob_max,path

# 观测样本o属于各个隐藏状态s的概率
def prob_O2S(model,o):
    M_O2S = model["M_O2S"]
    return M_O2S[:, int(o)]

# 构建一个参数参数已知的HMM模型
model_hmm1 = dict()
# 状态的初始分布
model_hmm1["pi"] = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
# 各个状态之间的转移关系
model_hmm1["A"] = np.array([ # 行表示当前状态，列表式下一状态
                            [0.4,0.3,0.3],
                            [0.3,0.4,0.3],
                            [0.3,0.3,0.4]
                            ])

# 观测样本与各个状态之间的概率映射关系矩阵
M_O2S = np.zeros([3,8])
M_O2S[0,:6] = 1.0/6.0
M_O2S[1,:4] = 1.0/4.0
M_O2S[2,:8] = 1.0/8.0
model_hmm1["M_O2S"] = M_O2S
# print(model_hmm1)

# 计算观测样本O属于状态S的概率的函数
model_hmm1["B"] = prob_O2S

# 测试
'''任务3 利用维特比算法使用'''
datas = np.array([5,7,7,5])
_,path = decoder(model_hmm1,datas)
print(path)
print(model_hmm1["A"][:,0])
print(model_hmm1["A"][:,1])
print(model_hmm1["A"][:,2])