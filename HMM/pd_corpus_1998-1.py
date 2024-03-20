import os

# 数据文件所在目录
base_dir = os.path.join(os.path.dirname(__file__),'nlp_data')

# 打开文件 获取所有行记录
with open(os.path.join(base_dir, 'pku_training.utf8'),'r', encoding='utf-8') as fh:
    lines = fh.read().split('\n')

# 查看训练样本记录
subl = lines[:5]
for line in subl:
    tokens = line.split(' ')
    tokens = list(filter(lambda x: x != '', tokens))
    print(tokens)