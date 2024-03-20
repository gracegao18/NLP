import torch
# from torchcrf import CRF
from torch_crf_code import CRF

num_tags = 5
model = CRF(num_tags)

seq_length = 3
batch_size = 2

# compute log likelihood
# 创建随机发射矩阵
emissions = torch.randn(seq_length, batch_size, num_tags)
# 一套对应的标签
tags = torch.tensor(
    [[0,1],
     [2,4],
     [3,1]], dtype=torch.long)
print(model(emissions, tags))

# padding use mask tensor
mask = torch.tensor(
    [[1,1],
     [1,1],
     [1,0]], dtype=torch.uint8)
print(model(emissions, tags, mask=mask))

# Decode：把发射矩阵预测出来的结果进行解码 => 预测结果
print(model.decode(emissions))