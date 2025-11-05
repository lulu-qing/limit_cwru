# import argparse

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.resnet18_encoder import *
# from models.resnet20_cifar import *


# class MYNET(nn.Module):

#     def __init__(self, args, mode=None):
#         super().__init__()

#         self.mode = mode
#         self.args = args
#         self.num_features = 64
#         if self.args.dataset in ['cifar100']:
#             self.encoder = resnet20()
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

#     def forward_metric(self, x):
#         x = self.encode(x)
#         if 'cos' in self.mode:
#             x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
#             x = self.args.temperature * x

#         elif 'dot' in self.mode:
#             x = self.fc(x)
#             x = self.args.temperature * x
#         return x

#     def encode(self, x):
#         x = self.encoder(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = x.squeeze(-1).squeeze(-1)
#         return x

#     def forward(self, input):
#         if self.mode != 'encoder':
#             input = self.forward_metric(input)
#             return input
#         elif self.mode == 'encoder':
#             input = self.encode(input)
#             return input
#         else:
#             raise ValueError('Unknown mode')

#     def update_fc(self,dataloader,class_list,session):
#         for batch in dataloader:
#             data, label = [_.cuda() for _ in batch]
#             data=self.encode(data).detach()

#         if self.args.not_data_init:
#             new_fc = nn.Parameter(
#                 torch.rand(len(class_list), self.num_features, device="cuda"),
#                 requires_grad=True)
#             nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
#         else:
#             new_fc = self.update_fc_avg(data, label, class_list)

#         if 'ft' in self.args.new_mode:  # further finetune
#             self.update_fc_ft(new_fc,data,label,session)

#     def update_fc_avg(self,data,label,class_list):
#         new_fc=[]
#         for class_index in class_list:
#             #print(class_index)
#             data_index=(label==class_index).nonzero().squeeze(-1)
#             embedding=data[data_index]
#             proto=embedding.mean(0)
#             new_fc.append(proto)
#             self.fc.weight.data[class_index]=proto
#             #print(proto)
#         new_fc=torch.stack(new_fc,dim=0)
#         return new_fc

#     def get_logits(self,x,fc):
#         if 'dot' in self.args.new_mode:
#             return F.linear(x,fc)
#         elif 'cos' in self.args.new_mode:
#             return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

#     def update_fc_ft(self,new_fc,data,label,session):
#         new_fc=new_fc.clone().detach()
#         new_fc.requires_grad=True
#         optimized_parameters = [{'params': new_fc}]
#         optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

#         with torch.enable_grad():
#             for epoch in range(self.args.epochs_new):
#                 old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
#                 fc = torch.cat([old_fc, new_fc], dim=0)
#                 logits = self.get_logits(data,fc)
#                 loss = F.cross_entropy(logits, label)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 pass

#         self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(32))
        self.feature.add_module('f_relu1', nn.ReLU())
        self.feature.add_module('f_pool1', nn.MaxPool1d(2, 2))

        self.feature.add_module('f_conv2', nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(64))
        self.feature.add_module('f_relu2', nn.ReLU())
        self.feature.add_module('f_pool2', nn.MaxPool1d(2, 2))

        self.feature.add_module('f_conv3', nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'))
        self.feature.add_module('f_bn3', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu3', nn.ReLU())
        self.feature.add_module('f_pool3', nn.MaxPool1d(2, 2))

        self.feature.add_module('f_conv4', nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'))
        self.feature.add_module('f_bn4', nn.BatchNorm1d(256))
        self.feature.add_module('f_relu4', nn.ReLU())
        self.feature.add_module('f_pool4', nn.MaxPool1d(2, 2))

        self.feature.add_module('f_conv5', nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same'))
        self.feature.add_module('f_bn5', nn.BatchNorm1d(512))
        self.feature.add_module('f_relu5', nn.ReLU())
        self.feature.add_module('f_pool5', nn.MaxPool1d(2, 2))
        # self.feature.add_module('f_flatten', nn.Flatten())


    def forward(self, input_data):
        
        feature = self.feature(input_data)      
#         feature = feature.view(-1, 512 * 4)

        return feature

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        self.num_features = 512
        if self.args.dataset in ['cwru']:
            self.encoder = CNNModel()
        
    #使用自适应池化层，将输出的特征图池化成一个大小为 (1, 1) 的特征图，这有助于将不同输入大小的特征图映射到相同的输出形状 
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))
        
        #定义一个全连接层 fc，将输入的 512 维特征映射到输出的类别数 num_classes
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        #如果模式是 'cos'，使用余弦相似度进行度量
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

       #如果模式是 'dot'，则使用点积方法
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        x = self.encoder(x)
        
        #对特征图进行自适应平均池化
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x


    def forward(self, input):
      #如果 self.mode != 'encoder'，则调用 forward_metric 进行度量学习的前向传播，返回处理后的 input  
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        
       #如果 self.mode == 'encoder'，则仅通过 encode 提取特征
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
        #将输入数据 data 通过模型的 encode 方法提取特征，返回的特征将不计算梯度    
            data=self.encode(data).detach()

    #如果 self.args.not_data_init 为 True，则初始化新的全连接层权重 new_fc，采用 kaiming_uniform_ 方法进行初始化
        if self.args.not_data_init: 
           #初始化一个新的可训练参数 new_fc         
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            #对 new_fc 进行 Kaiming 均匀初始化
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
            
        else:
            #计算每个类别的平均特征，并将其作为类别的原型（prototype），进而更新全连接层的权重
            new_fc = self.update_fc_avg(data, label, class_list)
            
#检查 self.args.new_mode 中是否包含 'ft'（表示 "finetuning"）,如果需要微调（finetune），则调用 update_fc_ft 方法对全连接层的权重进行进一步的微调
        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    
   #通过计算每个类别的平均特征（原型）来更新全连接层（FC）权重 
    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            # print(embedding.shape)
            proto=embedding.mean(0)
            # print(proto.shape)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
      
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

import torch
from torchsummary import summary
if __name__ == '__main__':
    x = torch.randn(size=(1,1,1024))
    # x = torch.randn(size=(1,64,224))
    # model = Bottlrneck(64,64,256,True)
    model = MYNET(args, mode=args.base_mode)
    # model = resnet20()
    # model = CNNModel()
    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    # print(model)
    summary(model,(1,1024), device='cpu')