import torch
import cv2
import numpy as np
import os
from os.path import join
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import sys

from dataset.transform import *

from PIL import Image

# 数据集
class SegDataset_severstal(Dataset):
    # 参数设置
    def __init__(self, root ='/root/autodl-tmp/MyFrame/datacode/Severstal_6666',split='val'
                 ,imagesize=300, cropsize=256,transform=None):
        self.root = root
        self.split=split
        lines = open(join(root,'{}.txt'.format(self.split)), 'r').readlines()
        self.samples=[]
        self.imagesize = imagesize
        self.cropsize = cropsize
        for line in lines:
            line = line.strip()
            self.samples.append(line)
        #训练集的数据增强
        self.train_augment = Compose([            
            HorizontalFlip(p=0.5),
            RandomScale((0.75, 1, 1.25)),
            RandomCrop((512,256))])
        #最后的Totensor和meanstd操作
        self.to_tensor = transform
    # 读取每个图片
    def __getitem__(self, index):
        name=self.samples[index]
        image = Image.open(join(self.root,'img','{}.jpg'.format(name)))
        label = Image.open(join(self.root,'label','{}.png'.format(name)))  ## 读成1通道
        #训练集数据增强
        if self.split=='train':
            im_lb = dict(im = image, lb = label)
            im_lb = self.train_augment(im_lb)
            image,label = im_lb['im'], im_lb['lb']
        image = self.to_tensor(image)
        label=np.array(label).astype(np.float32)
        # 返回图像及标签
        return image,label,image
    # 获取数据的容量
    def __len__(self):
        return len(self.samples)

    
if __name__ == '__main__':
        data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        data=SegDataset_severstal(transform=data_transform,split='train')
        print(len(data))