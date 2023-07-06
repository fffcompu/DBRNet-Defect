import numpy as np
import os
import cv2
import scipy.misc as m
import shutil


path = ''
dst_path = ''
imgs = os.listdir(path)
for img in imgs:
    format = img[-3:]
    if format == 'jpg':
        shutil.move(os.path.join(path,img),dst_path)





# def encode_segmap(mask):
#     """Encode segmentation label images as pascal classes
#     Args:
#         mask (np.ndarray): raw segmentation label image of dimension
#           (M, N, 3), in which the Pascal classes are encoded as colours.
#     Returns:
#         (np.ndarray): class map with dimensions (M,N), where the value at
#         a given location is the integer denoting the class index.
#     """
#     mask = mask.astype(int)
#     label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
#     for ii, label in enumerate(get_pascal_labels()):
#         label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
#     label_mask = label_mask.astype(int)
#     return label_mask
#
#
# def decode_segmap(label_mask, n_classes):
#     label_colours = get_pascal_labels()
#     r = label_mask.copy()
#     g = label_mask.copy()
#     b = label_mask.copy()
#     for ll in range(0, n_classes):
#         r[label_mask == ll] = label_colours[ll, 0]
#         g[label_mask == ll] = label_colours[ll, 1]
#         b[label_mask == ll] = label_colours[ll, 2]
#     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
#     rgb[:, :, 0] = r / 255.0
#     rgb[:, :, 1] = g / 255.0
#     rgb[:, :, 2] = b / 255.0
#     return rgb
#
# def get_pascal_labels():
#
#     return np.asarray(
#         [
#             [0, 0, 0],
#             [128, 0, 0],
#             [0, 128, 0],
#             [128, 128, 0],
#             [0, 0, 128],
#             [128, 0, 128],
#             [0, 128, 128],
#             [128, 128, 128],
#             [64, 0, 0],
#             [192, 0, 0],
#             [64, 128, 0],
#             [192, 128, 0],
#             [64, 0, 128],
#             [192, 0, 128],
#             [64, 128, 128],
#             [192, 128, 128],
#             [0, 64, 0],
#             [128, 64, 0],
#             [0, 192, 0],
#             [128, 192, 0],
#             [0, 64, 128],
#         ]
#     )
#
# path = '/media/aries/Udata/缺陷数据集/NEU_Seg-main/annotations/test'
# path_rgb = '/media/aries/Udata/缺陷数据集/NEU_Seg-main/annotations/test_rgb'
# imgs = os.listdir(path)
# for img in imgs:
#     img_path = img
#     img = cv2.imread(os.path.join(path,img),0)
#     print(img.shape)
#     img_rgb = decode_segmap(img,3)
#     print(img_rgb.shape)
#     m.imsave('/media/aries/Udata/缺陷数据集/NEU_Seg-main/annotations/test_rgb/{}'.format(img_path),img_rgb)
#     print('save success!')



