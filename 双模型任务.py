import numpy as np

# 观测样本o属于各个隐藏状态s的概率
def prob_O2S(model,o):
    M_O2S = model["M_O2S"]
    return M_O2S[:,int(o)]

# 根据一组概率分布生成一个样本
def gen_one_sample_from_Prob_list(Prob_list):
    N_segment = np.shape(Prob_list)[0]
    
    # 将[0,1]的区间分为N_segment段
    prob_segment = np.zeros(N_segment)
    # 例如：Prob_list = [0.3,0.3,0.4]
    #      Prob_segment_segment = [0.3,0.6,1]
    for i in range(N_segment):
        prob_segment[i] = prob_segment[i-1]+Prob_list[i]
    
    S = 0
    # 生成0,1之间的随机数
    data = np.random.rand()
    # 查看生成的数值位于哪个段中
    for i in range(N_segment):
        if data <= prob_segment[i]:
            S = i
            break
    return S

def gen_samples_from_HMM(model,N):
    M_O2S = model["M_O2S"]
    
    datas = np.zeros(N)
    stats = np.zeros(N)
    
    # 得到初始状态，并根据初始状态生成一个样本
    init_S = gen_one_sample_from_Prob_list(model["pi"])
    stats[0] = init_S
    
    # 根据初始状态，生成一个数据
    datas[0] = gen_one_sample_from_Prob_list(M_O2S[int(stats[0])])
    
    # 生成其他样本
    for i in range(1,N):
        # 根据前一状态，生成当前状态
        stats[i] = gen_one_sample_from_Prob_list(model["A"][int(stats[i-1])])
        # 根据当前状态生成一组数据
        datas[i] = gen_one_sample_from_Prob_list(M_O2S[int(stats[i])])
    return datas,stats

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
        s_next = beta[t+1]*model["B"](model,o[t+1])   # ---- 第二步：前向递推
        beta[t] = np.dot(s_next,model["A"].T)
    
    return beta

def backward(model,observations):
    o = observations
    
    # 计算后向概率
    beta = calc_beta(model,o)
    s_next = beta[0]*model["B"](model,o[0])           # ---- 第三步：最终
    prob_seq_b = np.dot(s_next,model["pi"])
    
    return np.log(prob_seq_b)


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



# 构建一个参数参数已知的HMM模型
model_hmm2 = dict()

# 状态的初始分布
model_hmm2["pi"] = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
# 各个状态之间的转移关系
model_hmm2["A"] = np.array([ # 行表示当前状态，列表式下一状态
                            [0.4,0.3,0.3],
                            [0.3,0.4,0.3],
                            [0.3,0.3,0.4]
                            ])

# 观测样本与各个状态之间的概率映射关系矩阵
M_O2S = np.zeros([3,8])
M_O2S[0,:6] = [0.8,0.02,0.02,0.02,0.02,0.02]
M_O2S[1,:4] = 1.0/4.0
M_O2S[2,:8] = 1.0/8.0
model_hmm2["M_O2S"] = M_O2S
# print(model_hmm1)

# 计算观测样本O属于状态S的概率的函数
model_hmm2["B"] = prob_O2S


# 分别用两个HMM模型生成两组数据序列
datas_hmm1,_ = gen_samples_from_HMM(model_hmm1,100)
datas_hmm2,_ = gen_samples_from_HMM(model_hmm2,100)

p_d1m1 = backward(model_hmm1,datas_hmm1)
p_d1m2 = backward(model_hmm2,datas_hmm1)

p_d2m1 = backward(model_hmm1,datas_hmm2)
p_d2m2 = backward(model_hmm2,datas_hmm2)

print("p_d1m1",p_d1m1)
print("p_d1m2",p_d1m2)
print("------------------")
print("p_d2m1",p_d2m1)
print("p_d2m2",p_d2m2)