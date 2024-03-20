import numpy as np

'''
后向算法：定义变量
        t时刻状态为i时，生成后继的样本序列的概率 记作：后向概率
'''
def calc_beta(model,observations):
    o = observations
    N_samples = np.shape(o)[0]
    N_stats = np.shape(model["pi"])[0]
    
    beta = np.zeros([N_samples,N_stats])              # ---- 第一步：初始化
    
    # 反向初始值
    beta[-1] = 1
    
    for t in range(N_samples-2,-1,-1):
        # 由t+1时刻的beta以及t+1时刻的观测值计算
        # t+1时刻的状态值
        # print(model["B"](model,o[t+1]))
        s_next = beta[t+1]*model["B"](model,o[t+1])   # ---- 第二步：前向递推
        # print(model["A"].T)
        beta[t] = np.dot(s_next,model["A"].T)
    
    return beta

def backward(model,observations):
    o = observations
    
    # 计算后向概率
    beta = calc_beta(model,o)
    s_next = beta[0]*model["B"](model,o[0])           # ---- 第三步：最终
    prob_seq_b = np.dot(s_next,model["pi"])
    
    return np.log(prob_seq_b)

# 观测样本o属于各个隐藏状态s的概率
def prob_O2S(model,o):
    M_O2S = model["M_O2S"]
    return M_O2S[:,int(o)]

# 测试

'''任务2 利用后向算法 计算一个生成一个序列的概率'''
# 主运行程序
if __name__ == '__main__':
    
    '''设计一个HMM模型'''
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
    
    datas = np.array([1,3,4])
    print("beta")
    print(calc_beta(model_hmm1,datas))
    print("---------------------")
    print("backward",backward(model_hmm1,datas))