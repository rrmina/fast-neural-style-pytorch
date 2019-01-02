import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Grammian
def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)     # (B, H*W, C)
    return  torch.bmm(x, x_t) # Broadcast Matrix Multiplication (B, C, C)

# Load image file
def load_image(path):
    # Images loaded as BGR
    img = cv2.imread(path)
    return img

# Show image
def show(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img/255).clip(0,1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()

# Preprocessing ~ Image to Tensor
def itot(img, max_size=None):
    # Rescale the image
    H, W, C = img.shape
    if (max_size==None):
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])    
    else:
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(max_size),
            transforms.ToTensor()
        ])
    
    # Subtract the means
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68],[1,1,1])
    tensor = normalize_t(itot_t(img)*255)
    
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

# Preprocessing ~ Tensor to Image
def ttoi(tensor):
    # Add the means
    ttoi_t = transforms.Compose([
        transforms.Normalize([-103.939, -116.779, -123.68],[1,1,1])])
    
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img
