import torch
from image_quality_assessment import PSNR, SSIM
from imgproc import *
import os

device = torch.device("cuda", 0)

# Initialize the sharpness evaluation function
psnr = PSNR(0, only_test_y_channel=True).to(device, non_blocking=True)
ssim = SSIM(0, only_only_test_y_channel=True).to(device, non_blocking=True)

# Initialize metrics
psnr_metrics = 0.0
ssim_metrics = 0.0

gt_image_dir = '../dataset/Set5/'
sr_image_dir = '../dataset/SR_SRGAN'

image_names = os.listdir(gt_image_dir)

for image_name in image_names:
    gt_image_path = os.path.join(gt_image_dir, image_name)
    sr_image_path = os.path.join(sr_image_dir, image_name)
    gt_tensor = preprocess_one_image(gt_image_path, device)
    sr_tensor = preprocess_one_image(sr_image_path, device)

    # cal metrics
    p = psnr(sr_tensor, gt_tensor).item()
    s = ssim(sr_tensor, gt_tensor).item()
    psnr_metrics += p
    ssim_metrics += s

avg_psnr = psnr_metrics / len(image_names)
avg_ssim = ssim_metrics / len(image_names)
print(f"PSNR: {avg_psnr:4.2f} [dB]\n"f"SSIM: {avg_ssim:4.4f} [u]")