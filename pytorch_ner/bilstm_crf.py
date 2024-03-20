import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    # 文本token序列转换为token_index张量
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# 用数值稳定的方法计算的对数和的exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.reshape(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True)

        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 状态转移概率矩阵，表示每种状态转移为另一种状态的得分
        # nn.Parameter生成的值 自带梯度，在模型进行梯度计算和后续更新的时候 会自动加入计算 => 用于模型更新的参数值
        # transaction -> data：5行5列，每行代表一个标签，每列代表当前标签转换到下一标签的概率
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 给两种不可能的状态转移赋值为-10000： 
        # 其它状态标签永远不可能转移到[START_TAG]标签
        # [STOP_TAG]标签永远不可能转移到其它状态的标签
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # lstm最后一层的输出
        self.hidden = self.init_hidden()

    # 单元/ 细胞记忆
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim//2),
                torch.randn(2, 1, self.hidden_dim//2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function

        # torch.full创建一个shape为(1,5) 初始值全部为-10000.的张量矩阵
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # [START_TAG]位置的值为0
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 记录上一个状态的得分(语句开始时，初始状态即上一个状态)，包含不同类别的分数
        forward_var = init_alphas

        # 便利整个句子
        for feat in feats:
            alphas_t = []  # 前向张量记录序列的步长
            for next_tag in range(self.tagset_size):
                # 广播发射矩阵的分数:不管之前的标签是什么，它都是一样的
                # (batch_size, num_tags, 1)
                emit_score = feat[next_tag].reshape(1, -1).expand(1, self.tagset_size)

                # trans_score中第i个位置记录的是从i转换到next_tag的分数
                # (batch_size, 1, num_tags)
                trans_score = self.transitions[next_tag].reshape(1, -1)

                # next_tag_var中第i个位置记录的是在进行log-sum-exp之前(i -> next_tag)的值
                next_tag_var = forward_var + trans_score + emit_score

                # 计算所有标签变量的是所有log-sum-exp。
                alphas_t.append(log_sum_exp(next_tag_var).reshape(1))
            # 结果通过torch.cat拼接成一维张量
            forward_var = torch.cat(alphas_t).reshape(1, -1)
        # 语句循环结束后再加上结束位置得分
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # 调整embedding层输出的张量维度 (seq_len,1,embedding_dim)
        embeds = self.word_embeds(sentence).reshape(len(sentence), 1, -1)
        # lstm所有时间步的输出和最后一次输出的hidden_state
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 调整lstm输出张量维度(seq_len,rnn_hidden_dim)
        lstm_out = lstm_out.reshape(len(sentence), self.hidden_dim)
        # 线性层运算推理
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 反向指针列表
        backpointers = []

        # 初始化基于对数空间的维特比向量
        # torch.full创建一个shape为(1,5) 初始值全部为-10000.的张量矩阵
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # [START_TAG]位置的值为0
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var记录的是：第i步时，i-1步的维特比变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 记录推理步骤的反向指针
            viterbivars_t = []  # 记录推理步骤的维特比变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]保存的是上一步标记i的viterbi变量，
                # 加上从标记i转换到下一个标记的分数。
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id) # 记录最大值的标记索引
                viterbivars_t.append(next_tag_var[0][best_tag_id].reshape(1))
            # 状态转移分数 + 发射矩阵的分数，结果保存为forward_var
            # 代表上一步计算的各状态的最大值
            forward_var = (torch.cat(viterbivars_t) + feat).reshape(1, -1)
            backpointers.append(bptrs_t)

        # 加上转移到结束标签的概率,计算路径的最后得分
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 按照反向指针解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签(不需要返回)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        # 反转
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # 从BiLSTM模型获取的发射矩阵得分
        feats = self._get_lstm_features(sentence)
        # 计算分子
        forward_score = self._forward_alg(feats)
        # 计算分母
        gold_score = self._score_sentence(feats, tags)
        # log(分子/分母)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 从BiLSTM模型获取的发射矩阵得分
        lstm_feats = self._get_lstm_features(sentence)

        # 根据给定的矩阵，找到最佳路径(维特比)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

if __name__ == '__main__':
    START_TAG = "<START>"   # 起始位置
    STOP_TAG = "<STOP>"     # 结束位置
    EMBEDDING_DIM = 5       # Embedding隐藏层维度
    HIDDEN_DIM = 4          # BiLSTM隐藏层维度 = 隐藏节点数 = 输出向量的维度

    # 训练数据：二维矩阵，第一行是词汇，第二行是每个词汇对应的标签
    # B - 实体起始标记、I - 实体内容标记、O - 非实体标记
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    # 创建词汇到索引的映射word_to_ix
    word_to_ix = {} 
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    # 状态标签
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    # 创建模型对象 
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    # 模型训练优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练前先进行一轮推理
    with torch.no_grad():
        # 提取第1个语料的token列表，转换为token_index列表张量
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        # 提取第1个语料的tag列表，转换为tag_index列表张量
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        # 打印模型预测的输出
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            300):  # 循环次数，这里是简单样本，正常情况下我们不需要300轮循环训练
        for sentence, tags in training_data:
            # Step 1. 清除累计的梯度值
            model.zero_grad()

            # Step 2. 获取我们输入网络的张量，在这里是token的索引值
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. 运行前向运算
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. 计算损失，梯度后通过optimizer更新模型参数
            loss.backward()
            optimizer.step()

    # 训练后的模型推理
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))