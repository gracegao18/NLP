import torch.nn as nn


word_dict = list('从百草园到三味书屋')

# 词向量大小(词表大小)
vsize = len(word_dict)
# 词向量维度(自定义)
vdim = 20

# sparse(): 稀疏
embedding = nn.Embedding(vsize, vdim, sparse=True)

# 打印Embedding层中的'weight'参数(里面包含所有的词向量)
for param in embedding.named_parameters():
    print(param)