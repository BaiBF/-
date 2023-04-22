import os
import cv2
import matplotlib.pyplot as plt
import torch
from image_quality_assessment import PSNR, SSIM
from imgproc import *

device = torch.device("cuda", 0)

# Initialize the sharpness evaluation function
psnr = PSNR(0, only_test_y_channel=True).to(device, non_blocking=True)
ssim = SSIM(0, only_only_test_y_channel=True).to(device, non_blocking=True)

gt_image_dir = '../dataset/Set5/'
cnn = '../dataset/SR_SRCNN'
gan = '../dataset/SR_SRGAN'
bic = '../dataset/SR_bic'
image_name = 'baby.png'

image = []
m = []

gt_image_path = os.path.join(gt_image_dir, image_name)
cnn_path = os.path.join(cnn, image_name)
gan_path = os.path.join(gan, image_name)
bic_path = os.path.join(bic, image_name)

gt_tensor = preprocess_one_image(gt_image_path, device)
bic_tensor = preprocess_one_image(bic_path, device)
cnn_tensor = preprocess_one_image(cnn_path, device)
gan_tensor = preprocess_one_image(gan_path, device)

image.append(cv2.cvtColor(cv2.imread(gt_image_path), cv2.COLOR_BGR2RGB))
image.append(cv2.cvtColor(cv2.imread(bic_path), cv2.COLOR_BGR2RGB))
image.append(cv2.cvtColor(cv2.imread(cnn_path), cv2.COLOR_BGR2RGB))
image.append(cv2.cvtColor(cv2.imread(gan_path), cv2.COLOR_BGR2RGB))


m.append('original(GT)')
name = [' ', 'Bicubic', 'SRCNN', 'SRGAN']
i = 1
for item in [bic_tensor, cnn_tensor, gan_tensor]:
    p = psnr(item, gt_tensor).item()
    s = ssim(item, gt_tensor).item()
    m.append(f"{name[i]}    PSNR: {p:4.2f} [dB]\n"f"SSIM: {s:4.4f} [u]")
    i += 1


f, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].imshow(image[0])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(m[0])
for i in range(1, 4):
    ax[i % 5].imshow(image[i])
    ax[i % 5].set_title(m[i])
    ax[i % 5].set_xticks([])
    ax[i % 5].set_yticks([])
plt.show()