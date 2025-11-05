import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
from copy import deepcopy

#这段代码的作用是生成每个任务的支持集任务ID和标签。它通过随机选择类别，结合预设的基础矩阵，生成每个任务中不同类别的样本ID，并与对应的标签一起返回。每个任务都包含 num_way 个类别，每个类别有 num_shot 个样本
#也就是为每个任务选择所要用到的类别和每个类别的样本索引
def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []

    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))

    return permuted_ids, permuted_labels


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1])).long()
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies

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



#定义了 MYNET 类，它继承自 Net 类，是一个用于几类分类任务（例如 Few-Shot Learning）的深度学习模型
class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)
     #获取模型特征的维度   
        hdim=self.num_features
#         #初始化一个多头自注意力层
#         #1：表示使用的头数。这里是单头自注意力。
          # hdim：每个头的维度，即特征维度。
          # dropout=0.5：设置 dropout 层的概率，防止过拟合
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5) 
        self.seminorm=True 
        
        #创建并初始化卷积神经网络（CNN）模型 CNNModel，可能是用于提取特征
        self.encoder = CNNModel()
        if args.dataset == 'cwru':
            self.seminorm=False
  
  #定义 split_instances 方法，用于生成任务（支持集和查询集）并按指定的方式分割样本      
    def split_instances(self, support_label, epoch):
        args = self.args
        #crriculum for num_way:
        total_epochs=args.epochs_base
        
        #Linear increment
        #self.current_way=int(5+float((args.sample_class-5)/total_epochs)*epoch)
        #Linear drop
        #self.current_way=int(args.sample_class-float((args.sample_class-5)/total_epochs)*epoch)
        #Equal
        #self.current_way=10
        #Random Sample
        #self.current_way=np.random.randint(5,args.sample_class)
        
        #当前的任务类别数为 args.sample_class
        self.current_way=args.sample_class

        permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, num_shot=args.sample_shot, num_way=self.current_way, num_class=args.sample_class)
        # 重新组织生成的任务实例，返回一个包含支持集索引和标签的元组 index_label
        index_label=(permuted_ids.view(args.num_tasks, args.sample_shot, self.current_way), torch.stack(permuted_labels))
        
        return index_label


#定义模型的前向传播方法
    def forward(self, x_shot, x_query=None, shot_label=None,epoch=None):
      #如果模式是 'encoder'，则仅进行特征编码
        if self.mode == 'encoder':
            x_shot = self.encode(x_shot)
            return x_shot
        
        #如果不是 'encoder' 模式，将支持集（x_shot）通过编码器进行编码，得到嵌入
        else:
            
            support_emb = self.encode(x_shot)
            query_emb = self.encode(x_query)
            #根据支持集标签和当前 epoch，生成任务实例的索引和标签
            index_label = self.split_instances(shot_label,epoch)
            logits = self._forward(support_emb, query_emb, index_label)
            return logits


#定义 _forward 方法，计算支持集和查询集之间的预测结果
    def _forward(self, support,query,index_label):
        
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        num_dim = support.shape[-1]  #num_dim 是支持集样本的特征维度（即每个样本的长度）
        # organize support data
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))

        #计算每个类别的代表性向量
        proto = support.mean(dim=1) # Ntask x NK x d
        #num_proto 是每个任务的类别数量，也就是每个任务的原型数量（即支持集中的类别数量）
        num_proto = proto.shape[1]     
        logit = []

        num_batch=1  ##表示一次处理一个任务
        num_proto=self.args.num_classes
        #即要分类的样本数量
        num_query=query.shape[0]
        #即每个样本的特征长度
        emb_dim = support.size(-1)
        query=query.unsqueeze(1)

     #用于处理 元学习（meta-learning）任务中的分类器训练过程
        for tt in range(num_task):            
            # 创建一个单位矩阵 全局掩码，反映了对全局信息的关注。
            #这通常是通过模型的某些权重或共享参数（例如，来自某一共享网络层的权重）计算得到的。
            global_mask = torch.eye(self.args.num_classes).cuda()  
            
            #获取当前任务对应的支持集标签
            whole_support_index = support_labels[tt,:]
            #通过 support_labels 中的索引将 global_mask 中对应类别的位置设为 0
            global_mask[:, whole_support_index] = 0
            
            # 使用 one_hot 函数将 whole_support_index 转换为 one-hot 编码，这样我们可以为每个任务构造一个本地掩码
            local_mask = one_hot(whole_support_index, self.args.num_classes)
            
            #计算当前任务的分类器权重矩阵：通过将全局权重 self.fc.weight.t() 和 global_mask 相乘得到全局分类器部分，再加上原型（proto[tt, :]）和本地掩码的乘积得到本地分类器部分，再将两者相加得到当前类别的分类器
            current_classifier = torch.mm(self.fc.weight.t(), global_mask) + torch.mm(proto[tt,:].t(), local_mask) 
            #转置分类器，使得其形状适合后续操作           
            current_classifier = current_classifier.t() #100*64
            current_classifier=current_classifier.unsqueeze(0)
            
            #加一个维度并展开分类器，使得它适配批次大小和查询集的样本数量
            current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            
            #将分类器展平
            current_classifier = current_classifier.view(num_batch*num_query, num_proto, emb_dim)
            
            #将 current_classifier 和 query 拼接在一起，形成一个形状为 [num_batch * num_query, num_proto + 1, emb_dim] 的张量
            combined = torch.cat([current_classifier, query], 1) # Nk x (N + 1) x d, batch_size = NK
            # 使用自注意力机制（self.slf_attn）对 combined 进行处理。这一步是为了在查询集和当前分类器之间学习更复杂的关系
            combined = self.slf_attn(combined, combined, combined)
            # print(combined.shape)
            # compute distance for all batches
            current_classifier, query = combined.split(num_proto, 1)
            
        
            if self.seminorm:
                #如果启用了 seminorm，则对 current_classifier 进行归一化，使用余弦距离计算
                #norm classifier
                current_classifier = F.normalize(current_classifier, dim=-1) # normalize for cosine distance
            
            #使用 批量矩阵乘法（bmm）计算查询集和分类器之间的相似度，并进行温度缩放    
                logits = torch.bmm(query, current_classifier.permute([0,2,1])) /self.args.temperature
                logits = logits.view(-1, num_proto)
            else:
                #使用 F.cosine_similarity 计算余弦相似度，并进行温度缩放
                logits=F.cosine_similarity(query,current_classifier,dim=-1)
                logits=logits*self.args.temperature

            
            logit.append(logits)
        logit = torch.cat(logit, 1)
        logit = logit.view(-1, self.args.num_classes)   

        return logit    #返回最终的分类预测结果 logit，它包含每个查询样本的类别得分
    

   #这个函数用于训练过程中更新分类器的权重，尤其是对于未见类别的类权重 
    def updateclf(self,data,label):
        support_embs = self.encode(data)  
        num_dim = support_embs.shape[-1]
        
        #proto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d
        #将 support_embs 重塑为形状 (5, -1, num_dim)，其中 5 是支持集样本数目（这里假设支持集是5个样本）
        proto = support_embs.reshape(5, -1, num_dim).mean(dim=0) # N x d

        #使用自注意力机制（self.slf_attn）处理 proto 和共享的 self.shared_key，生成一个新的嵌入 cls_unseen，该嵌入是未见类别的类表示
        cls_unseen, _, _ = self.slf_attn(proto.unsqueeze(0), self.shared_key, self.shared_key)
        #cls_unseen = F.normalize(cls_unseen.squeeze(0), dim=1)
        cls_unseen=cls_unseen.squeeze(0)

        #将生成的 cls_unseen 向量赋值到全连接层 self.fc 的权重中，更新未见类别的类权重
        self.fc.weight.data[torch.min(label):torch.max(label)+1]=  cls_unseen
        
    def forward_many(self, query):
        #cls_seen = F.normalize(self.fc.weight, dim=1) 
        
        # 定义批次大小。num_batch=1 表示只有一个批次  
        num_batch=1
        #：定义类别的数量 num_proto
        num_proto=self.args.num_classes
        
        emb_dim = query.size(-1)
        query=query.view(-1,1,emb_dim)
        #计算查询样本的数量 num_query
        num_query=query.shape[0]

        #从全连接层 self.fc 中获取权重，并对权重张量进行扩展
        current_classifier = self.fc.weight.unsqueeze(0)
        #即为每个查询样本生成每个类别的权重
        current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        current_classifier = current_classifier.view(num_batch*num_query, num_proto, emb_dim)
        
        combined = torch.cat([current_classifier, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        # 将 combined 切分为两个部分：current_classifier 和 query，其中 current_classifier 是每个类别的嵌入，query 是查询样本的嵌入
        current_classifier, query = combined.split(num_proto, 1)
       
        if self.seminorm:
            #norm classifier
            current_classifier = F.normalize(current_classifier, dim=-1) # normalize for cosine distance
            logits = torch.bmm(query, current_classifier.permute([0,2,1])) /self.args.temperature
            logits = logits.view(-1, num_proto)
        else:
            #cosine
            logits=F.cosine_similarity(query,current_classifier,dim=-1)
            logits=logits*self.args.temperature
        return logits


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head  #多头注意力的头数，即将输入的特征空间划分成多少个子空间进行并行处理
        self.d_k = d_k #输入和输出的特征维度
        self.d_v = d_v  #: 每个注意力头的值（value）的维度

#         #w_qs：用于将输入的 q（查询）映射到多个注意力头的查询向量。
          # w_ks：用于将输入的 k（键）映射到多个注意力头的键向量。
          # w_vs：用于将输入的 v（值）映射到多个注意力头的值向量
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        #初始化全连接层的权重，使用的是正态分布初始化方法
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        #创建一个 ScaledDotProductAttention 实例，采用了缩放的点积注意力
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        
        #定义了一个层归一化（Layer Normalization）模块，用于在每个子层输出后进行规范化
        self.layer_norm = nn.LayerNorm(d_model)
        
        #定义了一个全连接层 fc，用于将多头注意力的输出映射回 d_model 维度
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):
        #d_k, d_v, n_head: 模型参数中获取每个头的维度和头数
        #sz_b, len_q, len_k, len_v: 获取 q, k, v 的形状
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
       #将查询 q、键 k 和值 v 通过对应的全连接层 w_qs, w_ks, w_vs 进行变换，得到多个头的查询、键、值向 
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
     
     #使用 permute 调整维度顺序，使得 q, k, v 的形状变为 (n_head * batch_size, sequence_length, dimension)，方便后续的批处理
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        #调用 ScaledDotProductAttention 模块进行注意力计算
        output, attn, log_attn = self.attention(q, k, v)
        

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

from torchsummary import summary
if __name__ == '__main__':
    x1 = torch.randn(size=(16, 1, 1024)).cuda()
    x2 = torch.randn(size=(128, 1, 1024)).cuda()
    y = torch.LongTensor([0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0]).cuda()
    # model = Bottlrneck(64,64,256,True)
    model = MYNET(args, mode=args.base_mode).cuda()
    # model = resnet20()
    # model = CNNModel()
    output = model(x1, x2, y, 0)
    # print(f'输入尺寸为:{x1.shape, x2.shape,y.shape,}')
    # print(f'输出尺寸为:{output.shape}')
    # print(model)
    # summary(model,[(3,32,32), (3,32,32)], device='cpu')