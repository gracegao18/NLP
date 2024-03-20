# 模型训练
import os
import torch
import torch.optim as optim
from bilstm_crf import BiLSTM_CRF
from ner_dataset import NerDataSet
from preprocess import build_vocab, build_tag_dict, read_corpus, build_data_loader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_model(model, train_dl):
    # 模型训练优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    running_loss = 0.0
    loss_num = 0
    # 训练
    for epoch in range(3):  
        for i, (token_idx, tag_idx) in enumerate(train_dl):
            # Step 1. 清除累计的梯度值
            model.zero_grad()

            # Step 3. 运行前向运算
            out = model(token_idx)
            loss = model.loss(out, tag_idx)

            # Step 4. 计算损失，梯度后通过optimizer更新模型参数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:    # 每 batchsize * 100 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss/100))
                running_loss = 0.0
                writer.add_scalar('train_loss', running_loss/100, loss_num)
                loss_num += 1


if __name__ == '__main__':
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型参数
    EMBEDDING_DIM = 16       
    HIDDEN_DIM = 32
    BATCH_SIZE = 16
    
    # 语料文件
    train_file = os.path.abspath('./chinese_ner/msra_bio/msra_train_bio')
    test_file = os.path.abspath('./chinese_ner/msra_bio/msra_test_bio')
    tag_file = os.path.abspath('./chinese_ner/msra_bio/tags.txt')
    # train_file = os.path.join(os.path.dirname(__file__), 'msra_bio/msra_train_bio')
    # test_file = os.path.join(os.path.dirname(__file__),'msra_bio/msra_test_bio')
    # tag_file = os.path.join(os.path.dirname(__file__), 'msra_bio/tags.txt')
    
    # 创建训练用语料
    train_tokens, train_tags = read_corpus(train_file)
    test_tokens, test_tags = read_corpus(test_file)
    # 构建词汇表和tag字典
    vocab = build_vocab(train_tokens)
    tag_to_ix = build_tag_dict(tag_file)
    # 自定义Dataset
    train_ds = NerDataSet(train_tokens, train_tags, vocab, tag_to_ix)
    # 构建DataLoader
    train_dl = build_data_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # for tokens, labels in train_dl:
    #     print(len(tokens))
    #     print(len(labels))
    #     print(len(tokens[0]))
    #     print(len(tokens[15]))
    #     break
    
    # 创建模型
    model = BiLSTM_CRF(
        vocab_size = len(vocab), 
        embedding_dim = EMBEDDING_DIM,
        hidden_dim = HIDDEN_DIM,
        taget_size = len(tag_to_ix))
    model.to(device)

    train_model(model, train_dl)
    