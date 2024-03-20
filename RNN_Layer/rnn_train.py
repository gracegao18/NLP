import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
from rnn_model import *
 
# 检测可用设备
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(train_loader, test_loader, model):
    optimizer = torch.optim.Adam(rnn_cls.parameters(), lr = lr)   # optimize all parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    size = len(train_loader.dataset)
    for _ in range(epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):   
            b_x = Variable(x.reshape(-1, 28, 28).to(device))   # reshape x to (batch, time_step, input_size)
            b_y = Variable(y.to(device))               # batch y

            output = model(b_x)             # rnn 输出
            loss = loss_func(output, b_y)   # 交叉熵损失
            optimizer.zero_grad()           # 清理上一次更新的梯度值
            loss.backward()                 # 后向传播，计算梯度
            optimizer.step()                # 更新梯度值
            if step % 100 == 0:
                loss, current = loss.item(), step * len(b_x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # 每个epochs训练后测试
        test(test_loader, model, loss_func)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 模型设置为评估模式，代码等效于 model.train(False)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.reshape(-1, 28, 28))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    # 模型训练超参数
    epochs = 3           # 训练几轮
    batch_size = 64      # 每次训练样本批次大小
    hidden_size = 64    # rnn 模型隐藏层参数大小
    time_step = 28      # rnn 时间步数 / 图片高度
    input_size = 28     # rnn 每步输入值 / 图片每行像素
    number_layers = 1   # rnn 模型的层数
    num_class = 10      # 预测的类别数
    lr = 0.001           # 学习率

    # Mnist 手写数字
    train_data = dsets.MNIST(root = 'data_sets/mnist', #选择数据的根目录
                            train = True, # 选择训练集
                            transform = transforms.ToTensor(), #转换成tensor变量
                            download = True) # 从网络上download图片
 
    test_data = dsets.MNIST(root = 'data_sets/mnist', #选择数据的根目录
                            train = False, # 选择测试集
                            transform = transforms.ToTensor(), #转换成tensor变量
                            download = True) # 从网络上download图片
 
    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data)

    # 创建RNN推理模型
    rnn_cls = GRU_Classfication(input_size, hidden_size, number_layers, num_class)
    rnn_cls.to(device)
    print(rnn_cls)
 
    # training and testing
    train(train_loader, test_loader, rnn_cls)
