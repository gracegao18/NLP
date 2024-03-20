import torch
from torch.optim import optimizer
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# torch.unsqueeze方法用来扩充张量的维度，参数dim指定扩充的维度
# torch.inspace和numpy方法等效，用于生成1维的等差数列
# torch.linspace(-1, 1, 1000): 创建等差数列在[-1~1]区间
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
print(x)

# 对目标lable增加噪点(创建和x个数匹配的,均值为0的正态分组随机值)
# https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal
y = x.pow(2) + 0.2*torch.normal(torch.zeros(*x.size()))

# TensorDataset即自定义dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Net(torch.nn.Module):
    """自定义神经网络"""
    
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)    # 全联接隐藏层
        self.predict = torch.nn.Linear(20, 1)    # 全联接输出层
        
    def forward(self, x):
        x = F.relu(self.hidden(x))    # 激活函数
        x = self.predict(x)    # 线性输出
        
        return x
        
if __name__ == '__main__':
    # 为每个优化器创建一个net
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    net_AdamDelta = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam, net_AdamDelta]
    
    # different optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    opt_AdamDelta = torch.optim.Adadelta(net_AdamDelta.parameters())
    
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam, opt_AdamDelta]
    
    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], [], []]    # 记录training时不同神经网络的loss
    
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())    # loss recorder
                
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'AdamDelta']
    
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label = labels[i])
        
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()