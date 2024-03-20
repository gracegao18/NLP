import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class AlexNet(nn.Module):    # 训练AlexNet
    '''
    三层卷积， 三层全联接（应该是5层，由于图片是32*32，且为了效率，这里设了三层）
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(    # 输入 32*32*3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),    #（32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(    # 输入 16*16*6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),    #（16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(    # 输入 8*8*16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),    #（8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(    # 输入 4*4*64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),    #（4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(    # 输入 2*2*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),    #（2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    # (2-2)/2+1 = 1
        )
        
        # 全联接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(-1, 128)
        x = self.dense(x)
        
        return x
    
if __name__ == '__main__':
    net = AlexNet()
    input_img = torch.tensor(np.random.random((1, 3, 32, 32)), dtype=torch.float32)
    writer.add_graph(net, input_img)
    writer.close()