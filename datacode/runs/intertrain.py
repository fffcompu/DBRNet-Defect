# -*- encoding:utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
# import tqdm
import os
import shutil
import cv2
import argparse
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
from model.DBRNet import DBRNet
from torchstat import stat
import time

# from model.DFANet import DFANet

# from model.fast_scnn import FastSCNN
# from model.icnet import ICNet
# from model.fcn import FCN8s
# from model.espnet import ESPNetV2
# from model.unet import UNet
# from model.cgnet import Context_Guided_Network
# from model.dnlnet import NonLocalNet
# from model.stdc import BiSeNet
from dataset.neu import SegDataset
from dataset.kaggle import SegDataset_severstal
from dataset.mt import SegDataset_mt_processlittle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.metric import SegmentationMetric
from utils.csv_writer import csv_writer
# from utils.optimizer import OhemCELoss
from utils.lr_scheduler import *
from loss.loss import AffinityLoss
from tqdm import tqdm
from loss.OhemCELoss import OhemCELoss
# 参数定义
parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
# 模型选择
parser.add_argument('--model', type=str, default='DBRNet',
                    choices=['bisenet', 'dfanet', 'sfnet','deeplabv3+','ddrnet','lednet','DBRNet','erfnet','enet'],
                    help='model name (default: fcn8s)')
# 数据集
parser.add_argument('--dataset', type=str, default='kaggle',
                    choices=['mt', 'neu', 'kaggle','neu_900'],
                    help='dataset name (default: neu)')
# 输入图像尺寸
parser.add_argument('--base_size', type=int, default=512,
                    help='base image size')
# 裁剪尺寸
parser.add_argument('--crop_size', type=int, default=512,
                    help='crop image size')
# 加载线程数
parser.add_argument('--workers', '-j', type=int, default=4,
                    metavar='N', help='dataloader threads')
# batch size
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
# 总的epoch
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 50)')
# 学习率
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-3)')
# 动量
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
# 衰减率
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                    help='w-decay (default: 5e-4)')
# 学习率升温
parser.add_argument('--warmup_iters', type=int, default=0,
                    help='warmup iters')
# 升温因子
parser.add_argument('--warmup_factor', type=float, default=1.0 / 3,
                    help='lr = warmup_factor * lr')
# 升温方式
parser.add_argument('--warmup_method', type=str, default='linear',
                    help='method of warmup')
# cuda设置
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
# 存储路径
parser.add_argument('--img_dir', default=r'/root/autodl-tmp/MyFrame/datacode/NEU_Seg/',
                    help='Directory for saving checkpoint models')
# 在某个epoch保存
parser.add_argument('--save_epoch', type=int, default=15,
                    help='save model every checkpoint-epoch')
# log文件保存
parser.add_argument('--log_dir', default='../logs/',
                    help='Directory for saving checkpoint models')
# 在某个iter保存
parser.add_argument('--log-iter', type=int, default=10,
                    help='print log every log-iter')
# 验证
parser.add_argument('--val-epoch', type=int, default=1,
                    help='run validation every val-epoch')
#warmup设置为多少epoch
parser.add_argument('--warmup_epoch', type=int, default=10,
                    help='run validation every val-epoch')
args = parser.parse_args()

# 构建tensorboard可视化
writer = SummaryWriter(r'/root/autodl-tmp/MyFrame/datacode/tesnsorboard/{}/{}'.format(args.model, args.dataset))
# 相关超参数设置
batchsize = args.batch_size
epochs = args.epochs
imagesize = args.base_size
cropsize = args.crop_size
warmupepoch=args.warmup_epoch
# 图像存储路径
data_path = args.img_dir
#
if args.dataset == 'neu':
    num_class = 4
elif args.dataset == 'kaggle':
    num_class = 5
elif args.dataset == 'mt':
    num_class = 6
elif args.dataset == 'neu_900':
    num_class= 4
# 数据预处理
data_transforms = {}
data_transforms['train'] = None
# 图像分割数据集获取
if args.dataset == 'neu':
    train_dataset = SegDataset(r'/root/autodl-tmp/MyFrame/datacode/NEU_Seg/', imagesize, cropsize, split='train',
                                transform=data_transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_dataset = SegDataset(r'/root/autodl-tmp/MyFrame/datacode/NEU_Seg/', imagesize, cropsize, split='test',
                              transform=data_transforms['val'])
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

elif args.dataset == 'mt':
    train_dataset = SegDataset_mt_processlittle('/root/autodl-tmp/MyFrame/datacode/MT/', imagesize, cropsize, split='train',
                               transform=data_transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_dataset = SegDataset_mt_processlittle('/root/autodl-tmp/MyFrame/datacode/MT/', imagesize, cropsize, split='val',
                             transform=data_transforms['val'])
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True )
#severstal steel defect dataset
elif args.dataset == 'kaggle':
    data_transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.3436, 0.3436, 0.3436], [0.2039, 0.2039, 0.2039])])
    train_dataset = SegDataset_severstal(transform=data_transform_train,split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    data_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3406, 0.3406, 0.3406], [0.2072, 0.2072, 0.2072])])
    val_dataset = SegDataset_severstal(transform=data_transform_val,split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

elif args.dataset=='neu_900':
    dataset_rootdir=r'/root/autodl-tmp/MyFrame/datacode/NEU-Seg900/NEU-Seg/'
    train_dataset = SegDataset(dataset_rootdir, imagesize, cropsize, split='train',
                              transform=data_transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_dataset = SegDataset(dataset_rootdir, imagesize, cropsize, split='test',
                           transform=data_transforms['val'])
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

max_step = (len(train_dataset) // args.batch_size) * args.epochs
# 构建字典存储数据集
image_datasets = {}
image_datasets['train'] = train_dataset
image_datasets['val'] = val_dataset
dataloaders = {}
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader

#加速推理
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# 定义网络
if args.model == 'unet':
    net = UNet(n_channels=3, n_classes=num_class).cuda()
elif args.model == 'stdc':
    net = BiSeNet('STDCNet813', num_class).cuda()
elif args.model == 'dnl':
    net = NonLocalNet(num_class=num_class).cuda()
elif args.model == 'bisenet':
    net = BiSeNetV2(n_classes=num_class,aux_mode='eval').cuda()
elif args.model == 'dfanet':
    ch_cfg = [[8, 48, 96],
              [240, 144, 288],
              [240, 144, 288]]
    net = DFANet(ch_cfg, 64, num_classes=num_class).cuda()
elif args.model == 'enet':
    net = ENet(num_classes=num_class).cuda()
elif args.model == 'cgnet':
    net = Context_Guided_Network(classes=num_class).cuda()
elif args.model == 'icnet':
    net = ICNet(nclass=num_class).cuda()
elif args.model == 'fcn':
    net = FCN8s(nclass=num_class).cuda()
elif args.model == 'espnet':
    net = ESPNetV2(nclass=num_class).cuda()
elif args.model== 'lednet':
    net=LEDNet(num_classes=num_class).cuda()
elif args.model=='deeplabv3+':
    net=DeepLab(num_classes=num_class,downsample_factor=8).cuda()
elif args.model=='ddrnet':
    net=DDRNet(num_classes=num_class).cuda()
    #stat(net,(3,512,512))
elif args.model=='sfnet':
    net=SFNet('ResNet-18',num_classes=4).cuda()
    #stat(net,(3,512,512))
elif args.model=='DBRNet':
    net= DBRNet(num_classes=num_class).cuda()
elif args.model=='erfnet':
    net= ERFNet(num_classes=num_class).cuda()
elif args.model=='enet':
    net= ENet(num_classes=num_class).cuda()
#默认输出损失函数
criterion = OhemCELoss(thresh=0.7, lb_ignore=255)
if args.model == 'bisenet':
    # bisenetV2四个头部损失函数
    criteria_aux = [ OhemCELoss(thresh=0.7, lb_ignore=255)
                    for _ in range(4)]
elif args.model=='ddrnet':
    criteria_aux=OhemCELoss(thresh=0.7, lb_ignore=255)
elif args.model=='DBRNet':
    criteria_aux=OhemCELoss(thresh=0.7, lb_ignore=255)     
elif args.model=='enet':
    criteria_aux=OhemCELoss(thresh=0.7, lb_ignore=255)     

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# 学习率调整
lr_scheduler = WarmupPolyLrScheduler(optimizer, power=0.9, max_iter=max_step, warmup_iter=warmupepoch*len(train_dataloader),
                                     warmup_ratio=0.1, warmup='exp', last_epoch=-1)
# 训练过程主循环


best_Miou = 0
best_Miou_train = 0
for epoch in range(epochs + 1):
    # 输出当前epoch进度
    # print('Epoch {}/{}'.format(epoch, epochs - 1))
    # 根据不同的阶段判定模型的模式
    for phase in ['train','val']:
        if phase == 'train':
            net.train(True)
        else:
            net.train(False)
        # loss及acc置零  running_loss 每一个epoch输出的loss
        running_loss = 0.0
        # running_affinity_loss = 0.0
        running_accs = 0.0
        # 每一个iter输出的loss
        run_iter_loss = 0.0
        # 设置进度条
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(dataloaders[phase])), file=sys.stdout, bar_format=bar_format)
        # 设置每个迭代的dataloader
        dataloader = iter(dataloaders[phase])
        # 计数器置零
        n = 0
        # iter迭代（在当前数据集内迭代完成一次称为一个epoch）
        # 计算每一个epoch所用的metric
        metric = SegmentationMetric(num_class)
        for idx in pbar:
            # 读取图像和标签
            # print(dataloader.next())
            # imgs, labels,org = data[1]
            imgs, labels, org = dataloader.next()
            # 转cuda模式
            img, label, org = imgs.cuda().float(), labels.cuda().float(), org.cuda().float()
            # 输入至网络
            if phase == 'train':
                if args.model == 'bisenet':
                    output, *logits_aux = net(img)
                    # output,prior = net(img)
                    # 计算损失
                    loss = criterion(output, label.long())
                    loss_aux = [crit(lgt, label.long()) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss + sum(loss_aux)
                elif args.model == 'lednet':
                    output = net(img)
                    loss = criterion(output, label.long())
                elif args.model=='deeplabv3+':
                    output=net(img)
                    loss=criterion(output,label.long())
                elif args.model=='ddrnet':
                    output,logit_aux=net(img)
                    loss=criterion(output,label.long())
                    loss_aux=criteria_aux(logit_aux,label.long())
                    loss=loss+loss_aux
                elif args.model=='sfnet':
                    output=net(img)
                    loss=criterion(output,label.long())
                elif args.model=='DBRNet':
                    output,logit_aux=net(img)
                    loss=criterion(output,label.long())
                    loss_aux=criteria_aux(logit_aux,label.long())
                    loss=loss+loss_aux
                # loss_affinity = criterion_affinity(prior,label)
                # loss = loss + 0.2*loss_affinity
                elif args.model=='erfnet':
                    output=net(img)
                    loss=criterion(output,label.long())
                # 获取mask
                output_mask = output.cpu().data.numpy().copy()
            else:
                with torch.no_grad():
                    if args.model == 'bisenet':
                        net.eval()
                        output = net(img)[0]
                    elif args.model == 'lednet':
                        net.eval()
                        output = net(img)
                    elif args.model=='deeplabv3+':
                        net.eval()
                        output = net(img)
                    elif args.model=='ddrnet':
                        net.eval()
                        output = net(img)
                    elif args.model=='sfnet':
                        net.eval()
                        output = net(img)
                    elif args.model=='DBRNet':
                        net.eval()
                        output=net(img)
                    elif args.model=='erfnet':
                        net.eval()
                        output = net(img)
                    elif args.model=='enet':
                        net.eval()
                        output = net(img)
                loss = criterion(output, label.long())
                # 获取mask
                output_mask = output.cpu().data.numpy().copy()
            # 判定mask的最高得分
            output_mask = np.argmax(output_mask, axis=1)
            result = np.squeeze(output_mask)
            result = (result * 85).astype(np.uint8)
            # 标签值
            y_mask = label.cpu().data.numpy().copy()
            lbl_img = np.squeeze(y_mask)
            lbl_img = (lbl_img * 85).astype(np.uint8)
            # 原图
            org = org.cpu().data.numpy().copy()
            org = np.squeeze(org)
            # 计算精度
            acc = (output_mask == y_mask)
            # 求均值
            acc = acc.mean()
            # 训练阶段反向传播，并更新参数
            if phase == 'train':
                # 优化器清零
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            # 计算loss及acc
            running_loss += loss.data.item()
            # running_affinity_loss += loss_affinity.data.item()
            running_accs += acc
            run_iter_loss = loss.data.item()
            n += 1
            resultimage_p = org.copy()
            resultimage_l = org.copy()
            metric.addBatch(output_mask, y_mask)
            # 输出每一个iter的内容
            print_str = 'Epoch{}/{}'.format(epoch, epochs + 1) \
                        + ' Iter{}/{}:'.format(idx + 1, len(dataloaders[phase])) \
                        + ' loss_sup=%.5f' % run_iter_loss
            if(epoch!=epochs):
                        print_str=print_str+ '   lr=%.4e' % lr_scheduler.get_lr()[0]
            pbar.set_description(print_str, refresh=False)

        # calculate
        pa = metric.pixelAccuracy()
        Mpa, Cpa = metric.meanPixelAccuracy()
        Miou, Ciou = metric.meanIntersectionOverUnion()
        FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
        Precision = metric.precision()
        Recall = metric.recall()
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))
        MF = np.nanmean(F1)
        Iou = metric.IntersectionOverUnion()

        # per iou
        PerCiou_set = {}
        PerCiou = np.around(Ciou, decimals=num_class)
        for index, per in enumerate(PerCiou):
            PerCiou_set[index] = per
        # per class pa
        PerCpa_set = {}
        PerCpa = np.around(Cpa, decimals=num_class)
        for index, per in enumerate(PerCpa):
            PerCpa_set[index] = per

        # per F
        F_set = {}
        F1 = np.around(F1, decimals=num_class)
        for index, per in enumerate(F1):
            F_set[index] = per
        csv_path = '/root/autodl-tmp/MyFrame/datacode/output/{}/{}'.format(args.model, args.dataset)
        if os.path.exists(csv_path) == False:
            os.makedirs(csv_path)
        csv_writer(PerCiou_set, os.path.join(csv_path, 'ciou'))
        csv_writer(PerCpa_set, os.path.join(csv_path, 'cpa'))
        csv_writer(F_set, os.path.join(csv_path, 'f'))

        # 计算单个epoch的loss和均值
        epoch_loss = running_loss / n
        # epoch_affinity_loss = running_affinity_loss / n
        epoch_acc = running_accs / n
        # 向tensorborad中添加数据
        if phase == 'train':
            writer.add_scalar('trainloss', epoch_loss, epoch)
            # writer.add_scalar('data/trainaffinityloss', epoch_affinity_loss, epoch)
            writer.add_scalar('data/trainacc', epoch_acc, epoch)
            writer.add_scalar('data/trainpa', pa, epoch)
            writer.add_scalar('data/trainMpa', Mpa, epoch)
            writer.add_scalar('data/trainMiou', Miou, epoch)
            writer.add_scalar('data/trainMF', MF, epoch)
            print('train epoch_{} loss={}'.format(epoch, str(epoch_loss)))
            # print('train epoch_{} loss_affinity={}'.format(epoch,str(epoch_affinity_loss)))
            print('train epoch_{} acc={}'.format(epoch, str(epoch_acc)))
            print('train epoch_{} pa={}'.format(epoch, str(pa)))
            print('train epoch_{} Mpa={}'.format(epoch, str(Mpa)))
            print('train epoch_{} Miou={}'.format(epoch, str(Miou)))
            print('train epoch_{} CIou={}'.format(epoch, str(Ciou)))
            print('train epoch_{} FWIou={}'.format(epoch, str(FWIoU)))
            print('train epoch_{} MF={}'.format(epoch, str(MF)))
            is_best_train = Miou > best_Miou_train
            if is_best_train:
                best_Miou_train = Miou

            filename = 'checkpoint_train.pth'
            filename = os.path.join(csv_path, filename)
            torch.save(net, filename)
            if is_best_train:
                best_filename = 'best_checkpoint_train.pth'
                best_filename = os.path.join(csv_path, best_filename)
                shutil.copyfile(filename, best_filename)
            print('train epoch_{} best_mIoU={}'.format(epoch, str(best_Miou_train)))

        else:
            time_sum = 0
            writer.add_scalar('data/valloss', epoch_loss, epoch)
            writer.add_scalar('data/valacc', epoch_acc, epoch)
            writer.add_scalar('data/valpa', pa, epoch)
            writer.add_scalar('data/valMpa', Mpa, epoch)
            writer.add_scalar('data/valMiou', Miou, epoch)
            writer.add_scalar('data/valMF', MF, epoch)
            print('val epoch_{} loss={}'.format(epoch, str(epoch_loss)))
            print('val epoch_{} acc={}'.format(epoch, str(epoch_acc)))
            print('val epoch_{} pa={}'.format(epoch, str(pa)))
            print('val epoch_{} Mpa={}'.format(epoch, str(Mpa)))
            print('val epoch_{} Miou={}'.format(epoch, str(Miou)))
            print('val epoch_{} CIou={}'.format(epoch, str(Ciou)))
            print('val epoch_{} FWIou={}'.format(epoch, str(FWIoU)))
            print('val epoch_{} MF={}'.format(epoch, str(MF)))

            is_best = Miou > best_Miou
            if is_best:
                best_Miou = Miou

            filename = 'checkpoint.pth'
            filename = os.path.join(csv_path, filename)
            torch.save(net, filename)
            if is_best:
                best_filename = 'best_checkpoint.pth'
                best_filename = os.path.join(csv_path, best_filename)
                shutil.copyfile(filename, best_filename)
            print('val epoch_{} best_mIoU={}'.format(epoch, str(best_Miou)))
    # _0.001_448
    # # 每间隔5个epoch保存一次模型
    if epoch % 5 == 0:
        torch.save(net, os.path.join(csv_path, 'model_epoch_{}.pth'.format(epoch)))
        print('model_epoch_{}.pth saved!'.format(epoch))

