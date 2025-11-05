import torch
import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
# 实现了一种按照指定类别数量、每个类别内样本数量以及批次数量，从给定的有类别标签的数据集中进行分层采样的功能，用于训练的元学习任务
# 每次采样时具体抽到哪些类别是不确定的，具有随机性
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # 每次迭代（批次）的数量，控制返回批次的数量
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            #使用 torch.randperm 随机打乱类的索引，然后选择前 n_cls 个类的索引
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            for c in classes:
                l = self.m_ind[c]  # 获取类别 c 的所有样本索引
                pos = torch.randperm(len(l))[:self.n_per]  # 在该类别中随机选择 n_per 个样本的索引
                batch.append(l[pos])  #将选择的样本索引（即该类的随机样本）添加到当前批次中

            batch = torch.stack(batch).t().reshape(-1)
            # .t() 对堆叠的张量进行转置（交换行和列）
            # 里做转置的原因是为了在后续的 reshape 操作后，标签的顺序会是 abcdabcdabcd 的形式，而不是 aaaabbbbccccddddd
            yield batch
            # finally sample n_batch*  n_cls(way)* n_per(shot) instances. per bacth.


# 在采样过程中保留基础类别顺序的分层采样机制，按照给定的批次数量、每次采样的类别数量以及每个类别内的样本数量
# 保持类别原有顺序进行采样
#从给定的标签数据中以平衡的方式生成批次。具体来说，它允许按类别随机选择样本，每个批次包含固定数量的不同类别样本，从而确保在每个训练周期中，每个类别样本的均匀分布
class BasePreserverCategoriesSampler():
    #保base class 顺序
    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            classes=torch.arange(len(self.m_ind))
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
            # finally sample n_batch*  n_cls(way)* n_per(shot) instances. per bacth.

# 该采样器用于在少样本学习任务中，每个批次从不同类别中随机选取一部分样本
class NewCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per,):
        #label：是一个包含所有数据标签的数组或列表
        self.n_batch = n_batch  # 指定数据加载器（dataloader）中迭代的次数
        self.n_cls = n_cls    #：每个批次中要选择的类别数（即每个任务的类别数量）
        self.n_per = n_per  #从每个类别中选择的样本数量（即每个类别的样本数）

        label = np.array(label)  # all data label
        self.m_ind = []  # 存储每个类别的样本索引
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
    
    #创建一个从最小类别到最大类别的索引数组 classlist，它表示所有可能类别  
        self.classlist=np.arange(np.min(label),np.max(label)+1)
        #print(self.classlist)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.classlist:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # 从当前类别的样本索引 l 中随机选择 n_per 个样本，orch.randperm(len(l))：生成一个长度为 len(l) 的随机排列，表示随机排序所有样本的索引；[:self.n_per]：选取随机排列中的前 n_per 个样本索引
                
       #将当前类别的 n_per 个样本索引（l[pos]）添加到 batch 列表中         
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)  #合并张量
            
    #yield使得该方法成为生成器,每次调用时生成一个新的批次，直到迭代完成      
            yield batch
           

if __name__ == '__main__':
    # q=np.arange(5,10)
    # print(q)
    # y=torch.tensor([5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,5,5,55,])
    # label = np.array(y)  # all data label
    # m_ind = []  # the data index of each class
    # for i in range(max(label) + 1):
    #     ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
    #     ind = torch.from_numpy(ind)
    #     m_ind.append(ind)
    # print(m_ind, len(m_ind))
    # 假设我们有 6 个类别，每个类别有若干个样本
    labels = torch.tensor([0, 0, 1, 1, 3, 3, 4, 4, 5, 5, 2, 2])  # 12 个样本，6 个类别
    # sampler = CategoriesSampler(labels, n_cls=3, n_per=2, n_batch=1)
    # for idx in sampler:
    #     print(idx)  # 输出随机选择的样本索引

    # # 示例：按顺序选择每个类别，从每个类别中选择 2 个样本
    # sampler = BasePreserverCategoriesSampler(labels, n_cls=3, n_per=2, n_batch=1)
    # for idx in sampler:
    #     print(idx)  # 输出按顺序选择的样本索引

    # # 示例：按 classlist 顺序选择类别，从每个类别中选择 2 个样本
    sampler = NewCategoriesSampler(labels,n_batch=1,n_cls=1, n_per=2)
    for idx in sampler:
        print(idx)  # 输出按 classlist 顺序选择的样本索引
    # 示例数据集 (类别标签)
  
