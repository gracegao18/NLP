import torch
from torch import nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"

class RNNBasic_Classfication(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, num_class):
        super(RNNBasic_Classfication, self).__init__()

        self.number_layers = number_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNNCell(            # nn.RNNCell() 好多了
            input_size = input_size,      # 图片每行的数据像素点
            hidden_size = hidden_size     # rnn hidden unit
        )
        self.out = nn.Linear(hidden_size, num_class)    # 输出层
        
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # h_n shape (batch, hidden_size) rnn hidden
        h_n = Variable(torch.zeros(x.size(0), self.hidden_size).to(device))
        for i in range(x.size(1)):
            xt = x[:,i,:]
            h_n = self.rnn(xt, h_n)
        
        # 最后一个时间点的 hidden_out 输出
        out = self.out(h_n)
        return out

class RNN_Classfication(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(RNN_Classfication, self).__init__()
 
        self.rnn = nn.RNN(        # RNN模型
            input_size = input_size,      # 图片每行的数据像素点(特征数)
            hidden_size = hidden_size,     # rnn 隐藏层单元数
            num_layers = 1,       # 有几层 RNN layers
            batch_first = True,   # 指定batch为第一维 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, num_class)    # 输出层
        
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (batch, hidden_size) rnn hidden
        r_out, h_n = self.rnn(x)   
        
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:,-1,:])
        # out = self.out(h_n)
        return out

class LSTM_Classfication(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(LSTM_Classfication, self).__init__()
 
        self.rnn = nn.LSTM(         # LSTM模型
            input_size = input_size,      # 图片每行的数据像素点(特征数)
            hidden_size = hidden_size,     # rnn 隐藏层单元数
            num_layers = 1,       # 层数
            batch_first = True,   # 指定batch为第一维 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, num_class)    # 输出层
        
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) rnn hidden
        r_out, (c_n,h_n) = self.rnn(x)   
        
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:,-1,:])
        # out = self.out(h_n)
        return out

class GRU_Classfication(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(GRU_Classfication, self).__init__()
 
        self.rnn = nn.GRU(         # LSTM模型
            input_size = input_size,      # 图片每行的数据像素点(特征数)
            hidden_size = hidden_size,     # rnn 隐藏层单元数
            num_layers = 1,       # 层数
            batch_first = True,   # 指定batch为第一维 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, num_class)    # 输出层
        
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) rnn hidden
        r_out, h_n = self.rnn(x)   
        
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:,-1,:])
        # out = self.out(h_n)
        return out