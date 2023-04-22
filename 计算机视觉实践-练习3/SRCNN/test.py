import os
import PIL.Image as pil_image
from model import SRCNN
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_weights_path = './checkpoints/srcnn_x4.pth'
scale_factor = 4

model = SRCNN().to(device)

# Load the SRCNN weights
checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)
model.eval()

lr_image_dir = '../dataset/Set5_downscale_4'
sr_image_dir = '../dataset/SR_SRCNN'
sr_bic_dir = '../dataset/SR_bic'
image_names = os.listdir(lr_image_dir)

for image_name in image_names:
    lr_image_path = os.path.join(lr_image_dir, image_name)
    sr_image_path = os.path.join(sr_image_dir, image_name)
    sr_bic_path = os.path.join(sr_bic_dir, image_name)

    image = pil_image.open(lr_image_path).convert('RGB')

    # get bicubiced images
    image = image.resize((image.width * scale_factor, image.height * scale_factor), resample=pil_image.BICUBIC)
    image.save(sr_bic_path)
    image = np.array(image).astype(np.float32)

    # 对y通道做处理
    ycbcr = convert_rgb_to_ycbcr(image)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(sr_image_path)