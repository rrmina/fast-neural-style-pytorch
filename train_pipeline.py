import os
import sys
import random
import time
import logging
import torch
from torch import nn
import transformer
import torch.optim as optim

from loss_functions import ContrastiveLoss
from utils import (
    get_style_gram,
    itot,
    load_image,
    ttoi,
    saveimg, 
    plot_loss_hist
)
from train_utils import set_all_seeds

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s - %(asctime)s', datefmt='%d-%b-%y %H:%M:%S')
handler.setFormatter(formatter)
root.addHandler(handler)


class TrainPipeline:

    def __init__(
            self, 
            config,  
            hyper_param, 
            train_dataloader, 
            vgg, 
            use_one_style_image: bool = True
        ):

        self.config = config
        self.hyper_params = hyper_param
        set_all_seeds(seed=self.config.SEED)
        self._train_dataloader = train_dataloader
        self._initialize_model_params(vgg)
        self._initialize_loss_trackers()
        os.makedirs(self.config.SAVE_MODEL_PATH, exist_ok=True)
        os.makedirs(self.config.SAVE_IMAGE_PATH, exist_ok=True)
        self._batch_count = 1
        self._imagenet_neg_mean = torch.tensor(
            [-103.939, -116.779, -123.68], 
            dtype=torch.float32).reshape(1,3,1,1).to(self.config.DEVICE)

        
        self._style_image_paths = os.listdir(
                self.config.STYLE_IMAGE_PATH
            )
        if use_one_style_image:
            self._style_grams = self.__calculate_style_grams(
                self._style_image_paths[0]
            )
        self._style_grams = self._initialize_style_grams()
        self._style_image_indices = []

    def __calculate_style_grams(self, style_image_path):
        style_image = load_image(style_image_path)
        style_tensor = itot(style_image).to(self.config.DEVICE)
        style_tensor = style_tensor.add()
        B, C, H, W = style_tensor.shape
        style_features = self.vgg(self.style_tensor.expand(
            [self.hyper_params.BATCH_SIZE, C, H, W]))
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = get_style_gram(value)


    def _initialize_style_grams(self, style_image_paths):
        self.vgg.eval()
        style_grams = []
        with torch.no_grad():

            for image_path in style_image_paths:
                style_grams.append(self.__calculate_style_grams(image_path))
        

    def _initialize_loss_trackers(self):
        self._content_loss_history = []
        self._style_loss_history = []
        self._contrastive_loss_history = []
        self._total_loss_history = []
        self._batch_contrastive_loss_sum = 0 
        self._batch_content_loss_sum = 0
        self._batch_style_loss_sum = 0
        self._batch_total_loss_sum = 0


    def _initialize_model_params(self, vgg):
        # Load networks
        self.transformer_network = transformer.TransformerNetwork().to(self.config.DEVICE)
        # self.vgg = vgg.VGG16().to(self.config.DEVICE)
        self.vgg = vgg.to(self.config.DEVICE)

        # Optimizer settings
        self.optimizer = optim.Adam(
            self.transformer_network.parameters(), 
            lr=self.hyper_params.ADAM_LR
        )

        if self.hyper_params.USE_CONTRASTIVE_LOSS:
            # contrative loss to augment content similarity between styled and original image
            self.contrastive_loss_function = ContrastiveLoss(
                margin=self.hyper_params.CONTRASTIVE_MARGIN
            ).to(self.config.device)


    def train(self):

        if len(self._style_grams) == 1:
            style_gram = self._style_grams
            self._style_image_indices = [0]
        else:
            style_gram_index = random.randrange(len(self._style_image_indices))
            self._style_image_indices.append(style_gram_index)
            style_gram = self._style_grams[style_gram_index]

        for content_batch, _ in self._train_dataloader:
            # Get current batch size in case of odd batch sizes
            curr_batch_size = content_batch.shape[0]

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Zero-out Gradients
            self.optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch[:,[2,1,0]].to(self.config.DEVICE)
            generated_batch = self.transformer_network(content_batch)
            with torch.no_grad():
                content_features = self.vgg(content_batch.add(self._imagenet_neg_mean))
                generated_features = self.vgg(generated_batch.add(self._imagenet_neg_mean))

            # Content Loss
            MSELoss = nn.MSELoss().to(self.config.device)
            content_loss = self.hyper_params.CONTENT_WEIGHT * MSELoss(
                generated_features['relu2_2'], content_features['relu2_2'])            
            self._batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0
            for key, value in generated_features.items():
                s_loss = MSELoss(get_style_gram(value), style_gram[key][:curr_batch_size])
                style_loss += s_loss
            style_loss *= self.hyper_params.STYLE_WEIGHT
            self._batch_style_loss_sum += style_loss.item()

            if self.hyper_params.USE_CONTRASTIVE_LOSS:
                # Contrastive Loss
                contrastive_loss = self.config.CONTRASTIVE_WEIGHT * self.contrastive_loss_function(
                    anchor=content_features["relu2_2"], 
                    positive=generated_features["relu2_2"]
                )
                self._batch_contrastive_loss_sum += contrastive_loss.item()

                # Total Loss
                total_loss = content_loss + style_loss + contrastive_loss

            else:
                # Total Loss
                total_loss = content_loss + style_loss

            self._batch_total_loss_sum += total_loss.item()

            # Backprop and Weight Update
            total_loss.backward()
            self.optimizer.step()
        
    def training_loop(self):
        self.vgg.eval()
        start_time = time.time()
        for epoch in range(self.hyper_params.NUM_EPOCHS):
            logging.info(f"Epoch {epoch + 1}/{self.hyper_params.NUM_EPOCHS}")
            generated_batch, = self.train()
            # Save Model and Print Losses
            if ((
                (self._batch_count-1)%self.hyper_params.SAVE_MODEL_EVERY == 0) or 
                (self._batch_count==self.hyper_params.NUM_EPOCHS * len(self._train_dataloader))):
                # Print Losses
                logging.info(
                    f"Iteration {self._batch_count}/"
                    f"{self.hyper_params.NUM_EPOCHS*len(self._train_dataloader)}")
                logging.info(f"Content Loss: {self._batch_content_loss_sum/self._batch_count}")
                logging.info(f"Style Loss: {self._batch_style_loss_sum/self._batch_count}")
                if self.hyper_params.USE_CONTRASTIVE_LOSS:
                    logging.info(f"Contrasive Loss: {self._batch_contrastive_loss_sum/self._batch_count}")
                logging.info(f"Total Loss: {self._batch_total_loss_sum/self._batch_count}")
                logging.info(f"Time elapsed: {time.time()-start_time} seconds")

                # Save Model
                checkpoint_path = f"{self.config.SAVE_MODEL_PATH}checkpoint_{self._batch_count-1}.pth"
                torch.save(self.transformer_network.state_dict(), checkpoint_path)
                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = ttoi(sample_tensor.clone().detach())
                sample_image_path = (
                    f"{self.config.SAVE_IMAGE_PATH}sample0_{self._batch_count-1}.png"
                )
                saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                if self.hyper_params.USE_CONTRASTIVE_LOSS:
                    self._contrastive_loss_history.append(self._batch_contrastive_loss_sum/self._batch_count)
                self._content_loss_history.append(self._batch_total_loss_sum/self._batch_count)
                self._style_loss_history.append(self._batch_style_loss_sum/self._batch_count)
                self._total_loss_history.append(self._batch_total_loss_sum/self._batch_count)

            # Iterate Batch Counter
            self._batch_count+=1

        stop_time = time.time()

        final_path = self.config.SAVE_MODEL_PATH + "transformer_weight.pth"
        logging.info(f"Saving TransformerNetwork weights at {final_path}")
        torch.save(self.transformer_network.state_dict(), final_path)
        logging.info("Done saving final model")
        logging.info(f"Training Time: {stop_time-start_time} seconds")

        # Plot Loss Histories
        if (self.config.PLOT_LOSS):
            plot_loss_hist(
                self._contrastive_loss_history, 
                self._content_loss_history, 
                self._style_loss_history, 
                self._total_loss_history
            )

    @property
    def training_losses(self):
        return (
            self._contrastive_loss_history, 
            self._content_loss_history, 
            self._style_loss_history, 
            self._total_loss_history
        )