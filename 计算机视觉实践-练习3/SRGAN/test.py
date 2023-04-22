from model import *
from utils import *
import os

device = torch.device("cuda", 0)
g_model_weights_path = './checkpoints/SRGAN_x4-ImageNet-8c4a7569.pth.tar'

g_model = srresnet_x4(in_channels=3, out_channels=3,channels=64,num_rcb=16)
g_model = g_model.to(device)

# Load the super-resolution bsrgan_model weights
checkpoint = torch.load(g_model_weights_path, map_location=lambda storage, loc: storage)
g_model.load_state_dict(checkpoint["state_dict"])
g_model.eval()


lr_image_dir = '../dataset/Set5_downscale_4/'
sr_image_dir = '../dataset/SR_SRGAN'

image_names = os.listdir(lr_image_dir)

for image_name in image_names:
    lr_image_path = os.path.join(lr_image_dir, image_name)
    sr_image_path = os.path.join(sr_image_dir, image_name)
    lr_tensor = preprocess_one_image(lr_image_path, device)
    with torch.no_grad():
        sr_tensor = g_model(lr_tensor)

    # save image
    sr_image = tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(sr_image_path, sr_image)