import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import vgg
import transformer
import utils

# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "dataset"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "images/1-style.jpg"
BATCH_SIZE = 4 
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 5e0
TV_WEIGHT = 1e-6 
ADAM_LR = 0.001
SAVE_PATH = "transformer_weight.pth"

def train():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([103.939, 116.779, 123.68],[1,1,1]),
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load networks
    TransformerNetwork = transformer.TransformerNetwork().to(device)
    VGG = vgg.VGG16().to(device)

    # Get Style Features
    style_image = utils.load_image(STYLE_IMAGE_PATH)
    style_tensor = utils.itot(style_image).to(device)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)

    # Optimizer settings
    optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)

    for epoch in range (1, NUM_EPOCHS+1):
        print("Epoch {}/{}".format(epoch, NUM_EPOCHS+1))
        count = 0
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch.to(device)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch)
            generated_features = VGG(generated_batch)

            # Content Loss
            MSELoss = nn.MSELoss().to(device)
            content_loss = CONTENT_WEIGHT * MSELoss(content_features['relu4_3'], generated_features['relu4_3'])            

            # Style Loss
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(utils.gram(value), style_gram[key])
                style_loss += s_loss
            style_loss *= STYLE_WEIGHT

            # Total Loss
            total_loss = content_loss + style_loss

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    torch.save(TransformerNetwork.state_dict(), SAVE_PATH)

train()