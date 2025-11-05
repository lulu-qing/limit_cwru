from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
# x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.3,shuffle=True)

#用于加载和处理数据
class CWRU_dataset(Dataset):
    def __init__(self, root, train=True, index=None, base_sess=None,):
        if train:
            data = np.load(root + "/" + "train_data_0Hcwru_1_2048.npy")
            label = np.load(root + "/" + "train_label_0Hcwru_1_2048.npy")
        else:
            data = np.load(root + "/" + "test_data_0Hcwru_1_2048.npy")
            label = np.load(root + "/" + "test_label_0Hcwru_1_2048.npy")
        
       #将数据和标签转换为 PyTorch 的 tensor 类型
        self.data = torch.tensor(data, dtype=torch.float)
        self.targets = torch.tensor(label, dtype = torch.long)

        if base_sess:
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

        else:  # new Class session
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

    def __getitem__(self,index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)
    
    def SelectfromDefault(self, data, targets, index):
            data_tmp = []
            targets_tmp = []
            for i in index:
                #np.where(i == targets)[0]:找到所有标签等于 i 的数据的索引
                ind_cl = np.where(i == targets)[0]
                if data_tmp == []:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

            return data_tmp, targets_tmp

#用于从数据中选择新类别的样本  
    def NewClassSelector(self, data, targets, index):
            data_tmp = []
            targets_tmp = []
            ind_list = [int(i) for i in index]
            ind_np = np.array(ind_list)
            #ind_np.reshape((1,5)) 将 index 重塑为一个 1 行 5 列的数组，表示每个新类别的索引（表示一个类别的 5 个样本索引）
            index = ind_np.reshape((1,5))
            for i in index:
                ind_cl = i
                if data_tmp == []:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

            return data_tmp, targets_tmp
    
# class_index = np.arange(4)
# path = "./data"
# trainset = CWRU_dataset(root=path, train=True, index=class_index, base_sess=True)
