import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import argparse
from models.tcm import TCM



def image_to_tensor(image_path, target_size=(256, 256)):
    """
    Converts an image to a tensor and resizes it to the target size.

    Parameters:
    - image_path: Path to the image file.
    - target_size: A tuple (width, height) representing the desired output size.

    Returns:
    - A torch tensor representing the resized image.
    """
    image = Image.open(image_path).convert("RGB")
    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image_tensor = resize_transform(image)
    return image_tensor


def save_image_tensor(image_tensor, output_path):
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor)
    image.save(output_path)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process an image with a trained model.')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
parser.add_argument('--n', type=int, required=True, help='The dimension N to use for the model.')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the output image.')
args = parser.parse_args()

device = torch.device("cpu")  # Use "cuda" if GPU is available

# Load the checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)

# Initialize the model and load the state dict
net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.n, M=320)
net.to(device)
net.eval()
net.load_state_dict(checkpoint["state_dict"])
print('Checkpoint loaded from {args.checkpoint}')
# Process the image
image_tensor = image_to_tensor(args.image).to(device).unsqueeze(0)
print(image_tensor.shape)
# Assuming the model's output is an image tensor; adjust as needed
out_enc = net(image_tensor)#.squeeze(0)  # Dummy processing
print(out_enc['x_hat'])
out_dec = net.decompress(out_enc[1], (256,256))
# Save the output image
print(out_dec)
save_image_tensor(out_dec, args.save_path)

"""
python -u LIC_TCM/Compress.py --checkpoint /Users/jessegill/Desktop/MLP_CW4/output/0.0530_checkpoint.pth.tar --image /Users/jessegill/Desktop/MLP_CW4/data/kodim24.png --n 128 --save_path /Users/jessegill/Desktop/MLP_CW4/output/output_image.png

python -u LIC_TCM/eval.py --checkpoint /Users/jessegill/Desktop/MLP_CW4/output/0.0530_checkpoint.pth.tar --data /Users/jessegill/Desktop/MLP_CW4/data 

python -u LIC_TCM/train.py -d /Users/jessegill/Desktop/MLP_CW4/data --cuda --N 128 --lambda 0.05 --epochs 50 --lr_epoch 45 48  --save_path /Users/jessegill/Desktop/MLP_CW4/output/Normal


python -u LIC_TCM/eval.py --checkpoint /Users/jessegill/Desktop/MLP_CW4/output/Normal/0.0530_checkpoint.pth.tar --data /Users/jessegill/Desktop/MLP_CW4/data 


Normal transformers:
average_PSNR: 19.57dB
average_MS-SSIM: 4.8802
average_Bit-rate: 0.776 bpp
average_time: 1.110 ms


Hopfield:
average_PSNR: 19.44dB
average_MS-SSIM: 5.1721
average_Bit-rate: 0.672 bpp
average_time: 6.838 ms

Control: 
average_PSNR: 17.83dB
average_MS-SSIM: 3.7095
average_Bit-rate: 1.061 bpp
average_time: 0.788 ms

"""