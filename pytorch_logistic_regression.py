import torch
from sklearn.datasets import load_iris

# 只返回x，y值 不返回其他附加值
data, target = load_iris(return_X_y=True)

x = torch.tensor(data[50:150], dtype=torch.float32)    #指定输入x
# view(100, 1)：调整成张量，变成100行1列
y = torch.tensor(target[:100], dtype=torch.float32).view(100, 1)    # 指定输入y
# randn(1, 4, requires_grad=True)：随机正态分布，输出一行四类，requires_grad=True：包含计算梯度的隐藏值 
w = torch.randn(1, 4, requires_grad=True)    # 初始化参数w
b = torch.randn(1, requires_grad=True)    # 初始化参数b

learn_rate = 0.01    # 学习率
n_iters = 10000    # 最大迭代次数

for epoch in range(n_iters):
    # 前向计算
    y_ = torch.nn.functional.linear(input=x, weight=w, bias=b)    # 计算线性输出
    sy_ = torch.sigmoid(y_)    # 计算逻辑分布运算（输出的值可以作为概率使用）
    
    # 计算损失 ——> 损失函数：二分类的交叉熵 
    loss_mean = torch.nn.functional.binary_cross_entropy(sy_, y, reduction="mean")
    
    # backward：计算梯度
    loss_mean.backward()
    
    # 更新参数
    with torch.autograd.no_grad():    # 关闭梯度计算跟踪
        w -= learn_rate * w.grad    # 更新权重梯度
        w.grad.zero_()    # 清空本次计算的梯度（因为梯度是累积计算的，不清空就累加），zero_()：更新grad自身的值
        b -= learn_rate * b.grad    # 更新偏置梯度
        b.grad.zero_()    # 清空本次计算的梯度

    # 观察训练过程中的损失下降，与训练集预测的准确性
    if epoch % 500 == 0:
        print(F"误差损失值：{loss_mean:10.6f}，", end="")
        sy_[sy_ > 0.5] = 1
        sy_[sy_ <= 0.5] = 0

        correct_rate = (sy_ == y).float().mean()    # 逻辑值在Torch不给计算平均值，所以手动计算
        print(F"\t准确率为：{correct_rate*100: 8.2f}%")