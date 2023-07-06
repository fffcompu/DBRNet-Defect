import torch
import cv2 as cv
from torch.utils.data import Dataset
from torch.utils import data
import os

# 分类数据读取
class cls_dataset(Dataset):
   # 参数定义
   def __init__(self, txt_path= '/home/aries/Downloads/pretrain/list.txt',
                file_path='/home/aries/Downloads/pretrain/', transform=None):
       # txt路径
       self.txt_path = txt_path
       # 文件路径
       self.file_path = file_path
       # 打开txt文件
       fh = open(txt_path, 'r')
       # 构建图像及标签空列表
       imgs = []
       lbls = []
       # 列表内循环
       for line in fh:
           line = line.strip('\n')
           line = line.rstrip()
           words = line.split()
           imgs.append(words[0])
           lbls.append(words[1])
       self.imgs = imgs
       self.lbls = lbls
       self.transform = transform

   def __getitem__(self, index):
       img = self.imgs[index]
       lbl = self.lbls[index]
       img = cv.imread(os.path.join(self.file_path ,img))  # BGR
       img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
       if self.transform is not None:
           img = self.transform(img)
       return torch.from_numpy(img).float(), int(lbl)

   def __len__(self):
       return len(self.imgs)

# dataloader = cls_dataset(transform=None)
# train_dataset = data.DataLoader(dataloader, 1, shuffle=True,num_workers=4)
# print(len(dataloader))
# for iteration, (img,lbl) in enumerate(train_dataset):
#     print(img.shape)
#     print(lbl)


# # 数据增强后的训练集
# data_set = cls_dataset()
# train_size = int(0.8 * len(data_set))
# val_size = len(data_set) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(data_set, [train_size, val_size])
# train_dataset = data.DataLoader(train_dataset, 1, shuffle=True,num_workers=4)
# print(len(train_dataset))
# print(len(val_dataset))
