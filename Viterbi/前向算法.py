import numpy as np
'''
前向算法：定义变量
        已知前t个观测值，且第t个观测值属于状态i的概率
前向算法    alpha = [N_sample,N_stats]
表示已知前t个样本的情况下，第t个样本
属于状态i的概率
'''
def calc_alpha(model,observations):
    o = observations            # 观测序列长度
    N_samples = np.shape(o)[0]  # 观测样本数
    N_stats = np.shape(model["pi"])[0] # 状态数
    
    # alpha 初始化                                     ---- 第一步
    alpha = np.zeros([N_samples,N_stats])
    
    # 计算第0个样本属于第i个状态的概率
    alpha[0] = model["pi"]*model["B"](model,o[0])   # ---- 状态概率*观测概率
    
    # 计算其他时刻的样本属于第i个状态的概率              ---- 第二步 递推 求t=2,3,4,...T
    for t in range(1,N_samples):
        s_current = np.dot(alpha[t-1], model["A"])
        # s_current = alpha[t-1]*model["A"]
        # s_current = np.sum(s_current)
        alpha[t] = s_current*model["B"](model,o[t])
        
    return alpha

def forward(model,observations):
    o = observations
    
    # 计算前向概率
    alpha = calc_alpha(model,o)
    prob_seq_f = np.sum(alpha[-1])                  # ---- 第三步 最终
    
    return np.log(prob_seq_f)

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
    print("alpha")
    print(calc_alpha(model_hmm1,datas))
    print("---------------------")
    print("forward",forward(model_hmm1,datas))