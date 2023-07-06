import os
import torch

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# file_name = open('/home/aries/train.txt','a')
# images_path = '/media/aries/Udata/defect/NEU_Seg-main/training'
# images_path = os.path.abspath(images_path)
# images_path_contrast = images_path.split('/')[-1]
# print(images_path_contrast)
# # 查看文件夹下的图片
# images_name = os.listdir(images_path)
#
# count = 0
# # 遍历所有文件
# for eachname in images_name:
#     # 按照需要的格式写入目标txt文件
#     file_name.write(os.path.join(images_path_contrast,eachname) + ' '+ '16' + '\n')
#     count += 1
# print('生成txt成功！')
# print('{} 张图片地址已写入'.format(count))
# file_name.close()

# file_name = open('/home/aries/text.txt','a')
# images_path = '/media/aries/Udata/defect/NEU_Seg-main/images/test'
# images_path = os.path.abspath(images_path)
# print(images_path)
# images_path_contrast = images_path.split('/')[-1]
# print(images_path_contrast)
# # 查看文件夹下的图片
# images_name = os.listdir(images_path)

# count = 0
# # 遍历所有文件
# for eachname in images_name:
#     # 按照需要的格式写入目标txt文件
#     file_name.write(eachname.split('.')[0]+ '\n')
#     count += 1
# print('生成txt成功！')
# print('{} 张图片地址已写入'.format(count))
# file_name.close()
