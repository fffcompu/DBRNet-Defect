
import numpy as np
import torch.nn as nn
import torch
np.seterr(divide='ignore', invalid='ignore')
__all__ = ['SegmentationMetric']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def decode_segmap(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


'''
    /* Some formula about metric */
    (1) Confusion Matrix：    P     N      
                         P   TP    FP      
                         N   FN    TN      
        Some calculation in confusion matrix:
                         sum(axis=0) TP+FN
                         sum(axis=1) TP+FP
                         np.diag().sum() TP+TN
    (2) Precision = TP / (Tp+FP)
    (3) Recall = TP / (TP+FN)
    (4) Pixel Accuracy = (TP+TN) / (TP+TN+FP+FN)
    (5) Class Pixel Accuracy = 
    (6) IoU (Intersection of Union) = TP / (FP+TP+FN) 
    (7) mean IoU = 
    (8) Class IoU = 
    (9) FW IoU = 
'''

class SegmentationMetric(object):
    def __init__(self, numClass):
        # class number
        self.numClass = numClass
        # build zero matrix (num,num)
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def meanPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = TP / (TP + FP)
        # class pa
        Cpa = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # mean pa
        Mpa = np.nanmean(Cpa)
        return Mpa, Cpa

    def meanIntersectionOverUnion(self):
        # Intersection = TP ;Union = TP + FP + FN
        # cIoU = TP / (TP + FP + FN)
        # mIoU = cIoU/num
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        Ciou = (intersection / np.maximum(1.0, union))
        mIoU = np.nanmean(Ciou)
        return mIoU, Ciou
    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        #输出混淆矩阵
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIoU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def precision(self):
        # precision = TP / (TP+FP)
        precision = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        return precision

    def recall(self):
        # recall = TP / (TP+FN)
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)
        return recall

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict


        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        # the parameter is numpy array rather than tensor
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

# if __name__ == '__main__':
#     imgPredict = np.array([0, 0, 1, 1, 2, 2]) # 可直接换成预测图片
#     imgLabel = np.array([0, 0, 1, 1, 1, 2]) # 可直接换成标注图片
#     metric = SegmentationMetric(3) # 3表示有3个分类，有几个分类就填几
#     metric.addBatch(imgPredict, imgLabel)
#     pa = metric.pixelAccuracy()
#     mpa,cpa = metric.meanPixelAccuracy()
#     mIoU, per = metric.meanIntersectionOverUnion()
#     print('pa is : %f' % pa)
#     print('cpa is :{}'.format(cpa)) # 列表
#     print('mpa is : %f' % mpa)
#     print('mIoU is : %f' % mIoU, per)