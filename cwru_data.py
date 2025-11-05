# %%
from scipy.io import loadmat
import numpy as np
import os

file_names = os.listdir('E:\\数据集\\CWRU\\0HP')
print(file_names)
data_dict = []
for file_name in file_names:
    file_path = os.path.join('E:\\数据集\\CWRU\\0HP',file_name)
    file = loadmat(file_path)
    data=[]
    for key in file.keys():
        if 'DE'in key:
            data_dict.append(file[key])
# print(data_dict)                        
# print(data_dict.keys())

# %%
print(len(data_dict))
data_dict[0].shape

# %%
import matplotlib.pyplot as plt
# 滑窗窗口大小
window_size = 2048
# 滑窗步大小
step_size = 128

# data = np.array(data)
def sliding_window(arr,window_size, step_size):
    output = []
    for i in range(0, arr.shape[0] - window_size + 1, step_size):
        window = arr[i:i+window_size, :]
        output.append(window)
    return np.array(output)

df = []
# num = 3

for data in data_dict:

    output = sliding_window(data,window_size, step_size)[:400]

    print(output.shape)
    df.append(output)
#         df.append(output[:,:,num-1].reshape())
df = np.concatenate(df, axis=0).reshape(-1,1, window_size)
print(df.shape) 

n = len(df)//10
label = []
for i in range(10): 
    label.append(np.full(n,i))
label = np.concatenate(label,axis=0)
df = np.float32(df)

label = np.int64(label)

print(df.shape) 
print(np.unique(label))


# %%
# 用于存储训练集样本
train_data = []
train_labels = []

test_data = []
test_labels = []
# 对每一类进行处理
for class_label in range(10):
    # 找出属于当前类的样本索引
    class_indices = np.where(label == class_label)[0]
    
    # 取出每一类的前 300 个样本
    selected_indices = class_indices[:300]

    # 将这些样本添加到训练集
    train_data.append(df[selected_indices])
    train_labels.append(label[selected_indices])

    selected_indices1 = class_indices[300:]
    test_data.append(df[selected_indices1])
    test_labels.append(label[selected_indices1])
# 将列表中的样本和标签合并为 numpy 数组
train_data = np.vstack(train_data)
train_labels = np.concatenate(train_labels)

test_data = np.vstack(test_data)
test_labels = np.concatenate(test_labels)

print(f"训练集样本数: {train_data.shape}")
print(f"训练集标签数: {train_labels.shape}")
print(np.unique(train_labels))

print(f"测试集样本数: {test_data.shape}")
print(f"测试集标签数: {test_labels.shape}")
print(np.unique(test_labels))

# %%

np.save('train_data_0Hcwru_1_2048.npy', train_data)
np.save('train_label_0Hcwru_1_2048.npy', train_labels)

np.save('test_data_0Hcwru_1_2048.npy', test_data)
np.save('test_label_0Hcwru_1_2048.npy', test_labels)

# %%



