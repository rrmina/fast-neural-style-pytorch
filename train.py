from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np
import time

import vgg
import transformer
import utils

@dataclass
class ModelConfig:
    DATASET_PATH = "dataset"
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_LOSS = 1
    SEED = 35
    SAVE_IMAGE_PATH = "images/out/"
    SAVE_MODEL_PATH = "models/"
    STYLE_IMAGE_PATH = "images/mosaic.jpg"
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


def set_all_seeds(seed):
    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def transform_dataset(dataset_path, image_size, batch_size, transform=None):
    
    if not transform:
        # Dataset and Dataloader
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    train_dataset = datasets.ImageFolder(
        dataset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def train(config, hyper_params, train_loader):

    # Load networks
    TransformerNetwork = transformer.TransformerNetwork().to(config.DEVICE)
    VGG = vgg.VGG16().to(config.DEVICE)

    # Optimizer settings
    optimizer = optim.Adam(TransformerNetwork.parameters(), lr=hyper_params.ADAM_LR)

    # contrative loss to augment content similarity between styled and original image
    contrastive_loss_function = utils.ContrastiveLoss(
        margin=hyper_params.CONTRASTIVE_MARGIN
    ).to(config.device)

    # Loss trackers
    content_loss_history = []
    style_loss_history = []
    contrastive_loss_history = []
    total_loss_history = []
    batch_contrastive_loss = 0
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    # Get Style Features
    imagenet_neg_mean = torch.tensor(
        [-103.939, -116.779, -123.68], 
        dtype=torch.float32).reshape(1,3,1,1).to(config.DEVICE)
    style_image = utils.load_image(config.STYLE_IMAGE_PATH)
    style_tensor = utils.itot(style_image).to(config.DEVICE)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([hyper_params.BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)
    
    # Optimization/Training Loop
    batch_count = 1
    start_time = time.time()
    for epoch in range(hyper_params.NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch+1, hyper_params.NUM_EPOCHS))
        for content_batch, _ in train_loader:
            # Get current batch size in case of odd batch sizes
            curr_batch_size = content_batch.shape[0]

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch[:,[2,1,0]].to(config.DEVICE)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss
            MSELoss = nn.MSELoss().to(config.device)
            content_loss = hyper_params.CONTENT_WEIGHT * MSELoss(
                generated_features['relu2_2'], content_features['relu2_2'])            
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(utils.gram(value), style_gram[key][:curr_batch_size])
                style_loss += s_loss
            style_loss *= hyper_params.STYLE_WEIGHT
            batch_style_loss_sum += style_loss.item()

            # Contrastive Loss
            contrastive_loss = config.CONTRASTIVE_WEIGHT * contrastive_loss_function(
                anchor=content_features["relu2_2"], 
                positive=generated_features["relu2_2"]
            )
            batch_contrastive_loss += contrastive_loss.item()

            # Total Loss
            total_loss = content_loss + style_loss + contrastive_loss
            batch_total_loss_sum += total_loss.item()

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            # Save Model and Print Losses
            if ((
                (batch_count-1)%hyper_params.SAVE_MODEL_EVERY == 0) or 
                (batch_count==hyper_params.NUM_EPOCHS*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(
                    batch_count, hyper_params.NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(
                    batch_content_loss_sum/batch_count))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = config.SAVE_MODEL_PATH + "checkpoint_" + str(batch_count-1) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = utils.ttoi(sample_tensor.clone().detach())
                sample_image_path = config.SAVE_IMAGE_PATH + "sample0_" + str(batch_count-1) + ".png"
                utils.saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                contrastive_loss_history.append(batch_contrastive_loss/batch_count)
                content_loss_history.append(batch_total_loss_sum/batch_count)
                style_loss_history.append(batch_style_loss_sum/batch_count)
                total_loss_history.append(batch_total_loss_sum/batch_count)

            # Iterate Batch Counter
            batch_count+=1

    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history) 
    print("========Style Loss========")
    print(style_loss_history) 
    print("========Total Loss========")
    print(total_loss_history) 

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    final_path = config.SAVE_MODEL_PATH + "transformer_weight.pth"
    print("Saving TransformerNetwork weights at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(), final_path)
    print("Done saving final model")

    # Plot Loss Histories
    if (config.PLOT_LOSS):
        utils.plot_loss_hist(contrastive_loss_history, content_loss_history, style_loss_history, total_loss_history)