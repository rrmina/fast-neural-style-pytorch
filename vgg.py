import torch
import torch.nn as nn
from torchvision import models, transforms
import utils

class VGG19(nn.Module):
    def __init__(self, vgg_path="models/vgg19-d01eb7cb.pth"):
        super(VGG19, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg19_features = models.vgg19(pretrained=False)
        vgg19_features.load_state_dict(torch.load(vgg_path), strict=False)
        vgg19_features = vgg19_features.features

        # Turn-off Gradient History
        for param in vgg19_features.parameters():
            param.requires_grad = False

        # Reorganize layers ~ easier forward pass
        self.relu1_2 = torch.nn.Sequential(*list(vgg19_features.children())[0:4])
        self.relu2_2 = torch.nn.Sequential(*list(vgg19_features.children())[4:9])
        self.relu3_3 = torch.nn.Sequential(*list(vgg19_features.children())[9:18])
        self.relu4_2 = torch.nn.Sequential(*list(vgg19_features.children())[18:23])
        self.relu4_3 = torch.nn.Sequential(*list(vgg19_features.children())[23:27])
        self.relu5_3 = torch.nn.Sequential(*list(vgg19_features.children())[27:36])

    def forward(self, x):
        out_1_2 = self.relu1_2(x)
        h1, w1 = out_1_2.shape[3:4]

        out_2_2 = self.relu2_2(out_1_2)
        h2, w2 = out_2_2.shape[3:4]

        out_3_3 = self.relu3_3(out_2_2)
        h3, w3 = out_3_3.shape[3:4]

        out_4_2 = self.relu4_2(out_3_3)

        out_4_3 = self.relu4_3(out_4_2)
        h4, w4 = out_4_3.shape[3:4]

        out_5_3 = self.relu5_3(out_4_3)
        h5, w5 = out_5_3.shape[3:4]

        return [
            gram(out_1_2)/(h1*w1), # Style
            gram(out_2_2)/(h2*w2), 
            gram(out_3_3)/(h3*w3), 
            gram(out_4_3)/(h4*w4), 
            gram(out_5_3)/(h5*w5), 
            out_4_2                 # Content
            ]

def gram(x):
    return utils.gram(x)