import os
import torch
import numpy as np
from torchtext.datasets import SogouNews
from preprocess import build_vocab, build_data_loader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from rnn_classfication import TextClassficationModel
from cached_dataset import IterableCachedDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# 全局变量
EPOCHS = 2
BATCH_SIZE = 16
EMBEDDING_HIDDEN = 64
RNN_HIDDEN = 32

def model_train(model, trainloader, testloader):
    optimizer = SGD(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    total_size = len(trainloader)
    train_loss_cnt = 0
    test_acc_cnt = 0

    for epoch in range(EPOCHS):
        training_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            # 输入数据
            label, token = data
            token,label = token.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(token)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if i % 100 == 99 or i == total_size:
                print("epoch %d\t batch [%d/%d]\t loss:%.6f"%(epoch+1, i+1, total_size, training_loss / 100))
                writer.add_scalar('train loss of each 100 batch',training_loss / 100, train_loss_cnt)
                train_loss_cnt += 1
                training_loss = 0
            if i % 1000 == 999 or i == total_size:
                acc = test_accuracy(model, testloader)
                writer.add_scalar('test accuracy of each 1000 batch', acc, test_acc_cnt)
                test_acc_cnt += 1

def test_accuracy(model, testloader):
    total, correct = 0, 0
    with torch.no_grad():
        for data in testloader:
            token, label = data[1].to(device), data[0].to(device)
            out = model(token)
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        print('Accuracy of the network on test: %d %%' % (100 * correct / total))
        return 100.0 * correct / total
            
if __name__ == '__main__':
    model_file = os.path.join(os.path.dirname(__file__), 'trained_classfication.mod')
    
    # 加载数据集
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    train = SogouNews(root=data_path, split=('train'))
    # 构建词汇表和分词器
    tokenizer, vocab = build_vocab(train)
    
    # 重新加载数据集：这个数据集是一个iterable(可迭代)的dataset -> 迭代器 -> generator
    # generator：样本量过大(20w+)时，在数据加载时 可能在性能上有所损失，但是对内存空间使用更加科学合理。
    train,test = SogouNews(root=data_path, split=('train','test'))
    test_ds = IterableCachedDataSet(test)
    # 构建data_loader：可以多次使用，不用重新加载
    train_dl = build_data_loader(train, vocab, tokenizer, batch_size = BATCH_SIZE)
    test_dl = build_data_loader(test_ds, vocab, tokenizer, batch_size = BATCH_SIZE)

    if os.path.exists(model_file):
        model = torch.load(model_file)
    else:
        # 构建模型
        model = TextClassficationModel(
            len(vocab), 
            emb_hidden_size = EMBEDDING_HIDDEN,
            rnn_hidden_size = RNN_HIDDEN,
            num_layers = 1,
            num_class = 5)

    model.to(device)
    writer.add_graph(model,next(test_dl._get_iterator())[1])
    model_train(model, train_dl, test_dl)

    torch.save(model,model_file)