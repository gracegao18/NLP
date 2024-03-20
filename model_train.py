import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet

# 定义全局变量
modelPath = './model/alexnet_model.mod'
# 定义Summary_Writer,默认写入.run目录
writer = SummaryWriter()
# 训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型检验的频次
numPrint = 1000
# 模型训练批次数
batchSize = 10
# 模型训练轮数
epochs = 10

def load_data(batch_size):
    p = 1.0
    scale = (0.2, 0.3)
    ratio = (0.5, 1.0)
    value = (0, 0, 255)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value),
                transforms.ToPILImage()
            ])
    # 加载数据集 (训练集和测试集)
    # trainset = datasets.CIFAR10(root='cifar-10', train=True, download=True, transform=transform)
    trainset = datasets.CIFAR10(root='cifar-10', train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = datasets.CIFAR10(root='cifar-10', train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# 使用测试数据测试网络
def accuracy(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total

# 训练函数
def train(net, train_loader, test_loader):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降
    iter = 0
    num = 1
    # 训练网络
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            iter = iter + 1
            # 取数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
            # 将梯度置零
            optimizer.zero_grad()
            # 训练
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()   # 反向传播
            writer.add_scalar('train loss', loss.item(), iter)
            optimizer.step()  # 优化
            # 统计数据
            running_loss += loss.item()
            if i % numPrint == 999:    # 每 batchsize * 1000 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss/numPrint))
                running_loss = 0.0
                writer.add_scalar('accuracy', accuracy(test_loader), num)
                num += 1
    

if __name__ == '__main__':
    # 加载数据
    train_loader,test_loader = load_data(batchSize)

    # 创建或加载模型
    if os.path.exists(modelPath):
        net = torch.load(modelPath)
        print('模型加载成功！')
    else:
        net = AlexNet()
        print('模型创建成功!')
    print(f'使用{device}设备')
    net.to(device)

    # 模型训练
    train(net,train_loader, test_loader)
    writer.close()
    # 保存模型
    torch.save(net, modelPath)
    print('训练结束')