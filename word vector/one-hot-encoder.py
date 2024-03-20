import os
import torch

def one_hot_encoder_1(label_count):
    return torch.eye(label_count)

def one_hot_encoder_2(label_count):
    label = torch.range(0, label_count-1, dtype=int)
    label.unsqueeze_(-1)
    return torch.zeros(label_count, label_count).scatter_(1, label, 1)

if __name__ == '__main__':
    
    # 提取预料中的字，制作字典
    corpus_path = os.path.join(os.path.dirname(__file__), '从百草园到三味书屋.txt')
    
    char_set = set()
    for line in open(corpus_path, encoding='utf-8'):
        line = line.strip()
        if  line == '':
            continue
        char_set.update(set(line))
        
    char_dict = list(char_set)
    print(char_dict)
    
    # 通过字典制作one-hot编码
    one_hot = one_hot_encoder_2(len(char_dict))
    # 通过字符检索one-hot编码
    chr = input('请输入检索的字符：')
    idx = char_dict.index(chr)
    print(one_hot[idx, :])
    