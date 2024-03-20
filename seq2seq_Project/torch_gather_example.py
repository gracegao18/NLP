import numpy as np
import torch

# 3×3张量矩阵
m1 = torch.range(1,9).reshape(3,3)
print("矩阵内容:\n",m1.numpy())

# 索引
indices = torch.tensor([0,1]).reshape(1,-1)
print("索引值:\n", indices.numpy())
out1 = torch.gather(m1,0,indices)
print("检索dim=0结果:\n", out1.numpy())

out2 = torch.gather(m1,1,indices)
print("检索dim=1结果:\n", out2.numpy())
indices = torch.tensor([[0,1]])
print("索引值:\n", indices.numpy())
out2 = torch.gather(m1,1,indices)
print("检索dim=-1结果:\n", out2.numpy())
out2 = torch.gather(m1,-1,indices)

# 提取矩阵中对角值
ix = torch.tensor([[0,1,2]])
result = torch.gather(m1,0,ix)
print(result.numpy())

ix = torch.tensor([[0],[1],[2]])
result = torch.gather(m1,1,ix)
print(result.numpy())


