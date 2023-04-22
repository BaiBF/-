# 获取Bicubic插值下采样图像
import os
import cv2
import matplotlib.pyplot as plt


image_dir = "./dataset/Set5/"
# 获取文件夹中图片名

image_name = os.listdir(image_dir)
image = []
image_resize = []

save_dir = "./dataset/Set5_downscale_4/"
for i in range(len(image_name)):
    # image_name.append(image_path[i].split('/')[2][5:-4])
    image_path = image_dir + image_name[i]
    image.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    # 进行Bicubic插值下采样，缩小3倍
    image_resize.append(cv2.resize(image[i], None, fx=1/4, fy=1/4, interpolation=cv2.INTER_CUBIC))
    # cv2.imwrite(save_dir+image_name[i], cv2.cvtColor(image_resize[i], cv2.COLOR_RGB2BGR))

f, ax = plt.subplots(2, 5, figsize=(20, 10))
for i in range(5):
    ax[0, i % 5].imshow(image[i])
    ax[0, i % 5].set_title(image_name[i], y=-0.2)
    ax[1, i % 5].imshow(image_resize[i])
    ax[1, i % 5].set_title(image_name[i] + '_downscale_4', y=-0.2)
plt.show()