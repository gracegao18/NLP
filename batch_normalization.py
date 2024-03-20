import numpy as np
import torch
from torch import nn

# 原始数据
data = np.array([[1, 2],
                 [1, 3],
                 [1, 4]]).astype(np.float32)

# pytorch的BatchNorm1d
bn_torch = nn.BatchNorm1d(num_features=2)
data_torch = torch.from_numpy(data)
with torch.no_grad():
    bn_output_torch = bn_torch(data_torch)
    print(bn_output_torch.numpy())

def batch_norm(x, beta, gamma):
    """
    BN向传播
    :param x: 数据
    :return: BN输出
    """
    x_mean = x.mean(axis=0)
    x_var = x.var(axis=0)
    eps = 0.001
    # 对应论文中计算BN的公式
    x_hat = (x-x_mean)/np.sqrt(x_var+eps)
    y = gamma*x_hat + beta
    return y

with torch.no_grad():
    beta = bn_torch.bias.numpy()
    gamma = bn_torch.weight.numpy()
    bn_output = batch_norm(data, beta, gamma)
    print(bn_output)

print('-' * 40)

# affine=True(可训练参数)
m = nn.BatchNorm2d(2,affine=True)
print(m.weight)
print(m.bias)

input = torch.randn(1,2,3,4)
print(input)
output = m(input)
print(output)
print(output.size())
print('-' * 40)


# affine=False
m = nn.BatchNorm2d(2,affine=False)
print(m.weight)
print(m.bias)

input = torch.randn(1,2,3,4)
print(input)
output = m(input)
print(output)
print(output.size())