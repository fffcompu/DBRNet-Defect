import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from torchvision import transforms
import time
import cv2
import torch.nn.functional as F
import numpy as np
from utils.metric import AverageMeter,decode_segmap
import argparse
from model.DBRNet import DBRNet

parser = argparse.ArgumentParser(description='Semantic Segmentation Testing With Pytorch')
# 模型选择
parser.add_argument('--model', type=str, default='DBRNet',
                    choices=['bisenet','dfanet', 'bisenet','DBRNet'],
                    help='model name (default: fcn8s)')
# 数据集
parser.add_argument('--dataset', type=str, default='kaggle',
                    choices=['mt','neu','kaggle'],
                    help='dataset name (default: neu)')
args = parser.parse_args()
# 数据增强
data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.44545,0.44545,0.44545], [0.205742,0.205742,0.205742])])
data_transforms_anno = transforms.Compose([
            transforms.ToTensor(),])
# 预训练模型
modelpath = '/home/ubuntu//output/{}/{}/best_checkpoint_train.pth'.format(args.model,args.dataset)
# 载入模型
t_start = time.time()
net = torch.load(modelpath)
elapsed_time = time.time() - t_start
print("模型读取时间:{}".format(elapsed_time))
# 测试模式
net.eval()
# 测试图片路径
test_path = '/home/path'
anno_path = '/home/path'
# 测试图片名称列表
imagepaths = os.listdir(test_path)
# 停止autograd模块的工作，加速和节省显存
torch.no_grad()
fps = AverageMeter()
# 图片路径内循环
for imagepath in imagepaths:
    # 读取图像
    image = cv2.imread(os.path.join(test_path,imagepath))
    lbl = cv2.imread(os.path.join(anno_path,imagepath.split('.')[0]+'.png'),0)
    # resize图像至（244，244）
    image = cv2.resize(image,(448,448),interpolation=cv2.INTER_NEAREST)
    lbl = cv2.resize(lbl,(448,448),interpolation=cv2.INTER_NEAREST)
    # print(lbl.shape)
    lbl = decode_segmap(lbl)
    # 填充维度，从3维到4维
    imgblob = data_transforms(image).unsqueeze(0).cuda()
    # 初始时间
    t_start = time.time()
    # 获得原始网络输出，多通道
    predict= F.softmax(net(imgblob)).cpu().data.numpy().copy()
    # 测试时间
    elapsed_time = time.time() - t_start
    fps.update((1/elapsed_time))
    # 输出测试时间
    print('FPS: {fps.val:.3f} ({fps.avg:.3f})'.format(fps=fps))
    # 得到单通道label
    predict = np.argmax(predict, axis=1)
    # 降低维度，从4维到3维
    result = np.squeeze(predict)
    # 灰度拉伸，方便可视化
    result = decode_segmap(result)
    # 融合原始图像和预测结果图像
    combineresult = np.concatenate([image,result,lbl],axis=1)
    # combineresult = np.concatenate([image,result],axis=1)
    # 将图像保存至本地
    result_path = '/home/ubuntu//predict/show' #.format(args.dataset,args.model)
    if os.path.exists(result_path) == False:
            os.makedirs(result_path)
    cv2.imwrite(os.path.join(result_path, imagepath), combineresult)  # 写入新的目录



