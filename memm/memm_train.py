import os

# 初始状态概率
PROB_START = os.path.join(os.path.dirname(__file__), "data\prob_start.py")
# 发射矩阵(观测概率)
PROB_EMIT = os.path.join(os.path.dirname(__file__), "data\prob_emit.py")

def get_tag(token):
    """
    把词汇转换为BMES格式
    :param token: 输入的词汇
    :return BMES格式的文本
    """
    token_len = len(token)
    if token_len == 1:
        return 'S'
    m_count = token_len - 2
    return ''.join(['B', 'M' * m_count, 'E'])

def memm_train(lines):
    tags=["B","M","E","S"]
    o_tags=["B","M","E","S","start"]
    words_set = set()

    # 初始概率向量
    pre_a = {}
    line_number = 0   # 语料数
    all_words = []    # 每条语料中字的汇总
    all_tags = []     # 每条语料中[B,M,E,S]的标签汇总
    
    emit_C = {} # 发射矩阵
    count_tag = {}    # 统计所有样本中[B,E,M,S]各类别的数量

    for line in lines:
        line = line.strip()
        # 跳过空行
        if line == '':
            continue
        # 生成字的set()
        word_list = []
        line_number = line_number + 1
        word_list = [c for c in line if c != ' ']
        # 保存字典表
        words_set.update(set(word_list))
        # 保存语料中字的列表
        all_words.append(word_list)
        # 提取词汇
        words = line.split(" ")
        
        line_tags =[]
        for word in words:
            if word == '': 
                continue
            words_tags = get_tag(word)
            for tag in words_tags:
                line_tags.append(tag)
        # 保存语料中标签列表
        all_tags.append(line_tags)

    # 标签计数初始化
    for tag in tags:
        count_tag[tag] = 0

    # 发射矩阵初始化[S_t+1,S_t,O_t+1]
    for A in tags:
        word_dict = {}
        for B in o_tags:
            gailv = {}
            for wordA in words_set:
                gailv[wordA] = 0
            word_dict[B] = gailv
        emit_C[A] = word_dict

    #计算初始概率
    for line_tags in all_tags:
        lenght_tags=len(line_tags)
        if line_tags[0] in pre_a.keys():
            pre_a[line_tags[0]]=pre_a[line_tags[0]] + 1.0/line_number
        else:
            pre_a[line_tags[0]] = 1.0 / line_number
        count_tag[line_tags[0]]=count_tag[line_tags[0]]+1
    
    #不需要计算转移概率
    for line_tags in all_tags:
        lenght_tags = len(line_tags)
        for i in range(1,lenght_tags):
            count_tag[line_tags[i]]=count_tag[line_tags[i]]+1
            
    #计算给定标签下，观察值概率矩阵观察值是<St,Ot+1>而不是HMM的<Ot+1>
    for i in range(line_number):
        lenght=len(all_tags[i])
        for j in range(0,lenght):
            # S_t
            st = all_tags[i][j]
            # O_t+1
            ot1 = all_words[i][j]
            if j == 0:
                emit_C[st]["start"][ot1] = emit_C[st]["start"][ot1]+1.0
            else:
                emit_C[st][all_tags[i][j - 1]][ot1] = emit_C[st][all_tags[i][j - 1]][ot1] + 1.0
    
    for tag in tags:
        for key_pre in emit_C[tag].keys():
            for key_word in emit_C[tag][key_pre].keys():
                emit_C[tag][key_pre][key_word]=1.0*emit_C[tag][key_pre][key_word]/count_tag[tag]

    return pre_a, emit_C
    

if __name__=='__main__':
    corpus_file = os.path.join(os.path.dirname(__file__),'pku_training.utf8')
    with open(corpus_file,"r",encoding="utf8") as f:
        lines = f.readlines()

    pre_a, emit_C = memm_train(lines)

    with open(PROB_START, 'w', encoding='utf8') as start_fp:
        start_fp.write(str(pre_a))

    with open(PROB_EMIT, 'w', encoding='utf8') as emit_fp:
        emit_fp.write(str(emit_C))
    

