from dataclasses import dataclass
import torch

from torchvision import datasets, transforms
import numpy as np
import random


@dataclass
class ModelConfig:
    DATASET_PATH = "dataset"
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_LOSS = 1
    SEED = 35
    SAVE_IMAGE_PATH = "images/out/"
    SAVE_MODEL_PATH = "models/"
    STYLE_IMAGE_PATH = "images/"
    TRAIN_IMAGE_SIZE = 256


@dataclass
class TunableParameters:
    ADAM_LR = 0.001
    BATCH_SIZE = 4
    CONTENT_WEIGHT = 17 # 17
    CONTRASTIVE_MARGIN=1.0
    CONTRASTIVE_WEIGHT = 1e4
    NUM_EPOCHS = 1   
    STYLE_WEIGHT = 50 # 25
    SAVE_MODEL_EVERY = 500 # 2,000 Images with batch size 4
    USE_CONTRASTIVE_LOSS = True


def set_all_seeds(seed):
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataset(dataset_path, image_size, batch_size, transform=None):
    
    if not transform:
        # Dataset and Dataloader
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    
    dataset = datasets.ImageFolder(
        dataset_path, transform=transform)
    d_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    return dataset, d_loader
