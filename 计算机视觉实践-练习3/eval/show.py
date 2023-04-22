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
sr_image_dir = '../dataset/SR_SRGAN'
image_names = os.listdir(gt_image_dir)

gt_image = []
sr_image = []
m = []

for image_name in image_names:
    gt_image_path = os.path.join(gt_image_dir, image_name)
    sr_image_path = os.path.join(sr_image_dir, image_name)
    gt_tensor = preprocess_one_image(gt_image_path, device)
    sr_tensor = preprocess_one_image(sr_image_path, device)
    gt_image.append(cv2.cvtColor(cv2.imread(gt_image_path), cv2.COLOR_BGR2RGB))
    sr_image.append(cv2.cvtColor(cv2.imread(sr_image_path), cv2.COLOR_BGR2RGB))
    # cal metrics
    p = psnr(sr_tensor, gt_tensor).item()
    s = ssim(sr_tensor, gt_tensor).item()

    m.append(f"PSNR: {p:4.2f} [dB]\n"f"SSIM: {s:4.4f} [u]")

f, ax = plt.subplots(2, 5, figsize=(20, 10))
for i in range(5):
    ax[0, i % 5].imshow(gt_image[i])
    # ax[0, i % 5].set_title(image_name[i], y=-0.2)
    ax[1, i % 5].imshow(sr_image[i])
    ax[1, i % 5].set_title(m[i])
plt.show()