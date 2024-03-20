import torch
import torch.nn as nn
from torch.nn.modules.module import _forward_unimplemented
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from LeNet_pytorch import LeNet


# 训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型检验的频次
numPrint = 500
# 模型训练批次数
batch_size = 32
# 模型训练轮数
epochs = 10
# 学习率
learn_rate=0.001

def load_data(batch_size):
    # 图像预处理增强
    transform = transforms.Compose([
                transforms.ToTensor(),
                # 归一化
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # 随机水平翻转
                transforms.RandomHorizontalFlip(),
                # 随机垂直翻转
                transforms.RandomVerticalFlip(),
                # 随机倾斜15～30角度，其余区域用黑色填充
                transforms.RandomRotation(degrees=(15, 30), fill=(0, 0, 0)), 
            ])
    # 加载数据集：训练集：50000张训练图片
    trainset = torchvision.datasets.CIFAR10(root='cifar-10', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='cifar-10', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
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
    
    return 100.0 * correct / total

# 训练函数
def train(trainloader, testloader):
    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=learn_rate)  # 随机梯度下降
    
    # 训练网络
    for epoch in range(epochs):
        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
            # 取数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU
            # 将梯度置零
            optimizer.zero_grad()
            # 训练
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()   # 反向传播
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
            if step % numPrint == 499:    # 每 batchsize * 1000 张图片，打印一次
                acc = accuracy(testloader)
                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' % 
                      (epoch + 1, step + 1, running_loss/numPrint, acc))
                
    print('训练完成')
                
    

if __name__ == '__main__':
    # 加载数据
    trainloader,testloader = load_data(batch_size)

    # 创建或加载模型
    net = LeNet()
    net.to(device)

    # 训练模型
    train(trainloader, testloader)
    
    # 保存模型
    save_path = 'model/LeNet.mod'
    torch.save(net, save_path)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')