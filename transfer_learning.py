import torch
from torch import nn
from torchvision import models
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载预训练模型
# resnet_model = models.resnet18(pretrained=True)
# print(resnet_model)

# # 冻结层中参数
# for parma in resnet_model.parameters():
#     parma.requires_grad = False

# num_fc_in = resnet_model.fc.in_features
# # 改变全连接层，2分类问题，out_features = 2
# resnet_model.fc = nn.Linear(num_fc_in, 2)
# # 模型迁移到CPU/GPU
# model = resnet_model.to(device)
# # 定义损失函数
# loss_fc = nn.CrossEntropyLoss()
# # 选择优化方法
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


#######################################################################

vgg_model = models.vgg16(pretrained=True)
print(vgg_model)


for parma in vgg_model.parameters():
    parma.requires_grad = False

vgg_model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

vgg_model.to(device)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg_model.classifier.parameters())