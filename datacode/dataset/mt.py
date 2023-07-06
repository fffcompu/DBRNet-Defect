import torch
import cv2
import numpy as np
from os.path import join
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import random
import collections
# 数据集
class SegDataset_mt(Dataset):
    # 参数设置
    def __init__(self, root ='/home/ubuntu/metal_segmentation/MT',
                 imagesize=300, cropsize=256, split='train',transform=None):
        self.split = split
        lines = open(join(root,'{}.txt'.format(self.split)), 'r').readlines()
        self.samples = []
        self.root = root
        self.imagesize = imagesize
        self.cropsize = cropsize
        if transform is not None:
            self.transform = transform
                # 获取标签文件中的内容
        for line in lines:
            line = line.strip()
            self.samples.append(line)
    # 读取每个图片
    def __getitem__(self, index):
        name = self.samples[index]
        image = cv2.imread(join(self.root,'images','{}.jpg'.format(name)))
        label = cv2.imread(join(self.root,'annotations','{}.png'.format(name)), 0)  ##读成1通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#将cv2读取的BGR转换为RGB
        # resize处理
        image = cv2.resize(image, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        PIL_image = Image.fromarray(image)
        # offsetx = np.random.randint(self.imagesize - self.cropsize)
        # offsety = np.random.randint(self.imagesize - self.cropsize)
        # image = image[offsety:offsety + self.cropsize, offsetx:offsetx + self.cropsize]
        # label = label[offsety:offsety + self.cropsize, offsetx:offsetx + self.cropsize]
        # 返回图像及标签
        return self.transform(PIL_image), label,image
    # 获取数据的容量
    def __len__(self):
        return len(self.samples)

# data_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
# # Create Dataset
# trainset =SegDataset(transform=data_transform)
# # Create Training Loader
# train_dataset = DataLoader(trainset, 1, shuffle=True,num_workers=4)
# for iteration, (img,lbl,_) in enumerate(train_dataset):
#     print(img.shape)
#     print(lbl.shape)
class SegDataset_mt_processlittle(Dataset):
    def __init__(self, root ='/media/aries/Udata/defect/NEU_Seg-main',
                 imagesize=300, cropsize=256, split='test',transform=None):
        self.split = split
        lines = open(join(root,'{}.txt'.format(self.split)), 'r').readlines()
        self.samples = []
        self.root = root
        self.imagesize = imagesize
        self.cropsize = cropsize
        self.img_mean=[0.4366, 0.4366, 0.4366]
        self.img_std=[0.2112, 0.2112, 0.2112]
        if transform is not None:
            self.transform = transform
                # 获取标签文件中的内容
        for line in lines:
            line = line.strip()
            self.samples.append(line)
    # 读取每个图片
    def __getitem__(self, index):
        name = self.samples[index]
        image = cv2.imread(join(self.root, 'images', '{}.jpg'.format(name)))
        label = cv2.imread(join(self.root, 'annotations', '{}.png'.format(name)), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #eval直接返回
        if  self.split=='test'or self.split=='val':
            image = cv2.resize(image, (self.cropsize, self.cropsize), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.cropsize, self.cropsize), interpolation=cv2.INTER_LINEAR)
            return self.transform(image),label,image
        # resize处理
        image = cv2.resize(image, (self.imagesize, self.imagesize), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_LINEAR)
        # 使用随机缩放
        train_scale_array = [0.75, 1, 1.5, 1.75, 2.0]  # 随机缩放的尺寸大小
        scale = random.choice(train_scale_array)
        sh = int(image.shape[0] * scale)
        sw = int(image.shape[1] * scale)
        image = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        image = normalize(image, self.img_mean, self.img_std)
        if label is not None:
            label = cv2.resize(label, (sw, sh), interpolation=cv2.INTER_LINEAR)
        #对图片进行裁剪
        crop_size = (self.cropsize, self.cropsize)#获取裁剪的尺寸
        crop_pos = generate_random_crop_pos(image.shape[:2], crop_size)#开始裁剪的位置
        p_img, _ = random_crop_pad_to_shape(image, crop_pos, crop_size, 0)
        if label is not None:
            p_gt, _ = random_crop_pad_to_shape(label, crop_pos, crop_size, 255)
        # 返回图像及标签
        p_img = p_img.transpose(2, 0, 1)#转为CHW格式
        return torch.tensor(p_img), p_gt,p_img
    # 获取数据的容量
    def __len__(self):
        return len(self.samples)


#额外的数据处理
def get_2dshape(shape, *, zero=True):
    if not isinstance(shape, collections.Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape
def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w
def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                             border_mode, value=value)
    return img, margin
def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
               start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin
def normalize(img, mean, std):
    #uint转float32
    img = img.astype(np.float32) / 255.0
    #numpy-list相减变为float64
    img = img - mean
    img = img / std
    img=img.astype(np.float32)
    return img
    