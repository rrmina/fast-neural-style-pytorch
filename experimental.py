import torch
import torch.nn as nn

class TransformerNetworkV2(nn.Module):
    """
    Feedforward Transformation NetworkV2
    
        - No Tanh
        + Using Fully Pre-activated Residual Layers 
    """
    def __init__(self):
        super(TransformerNetworkV2, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayerV2(128, 3),
            ResidualLayerV2(128, 3),
            ResidualLayerV2(128, 3),
            ResidualLayerV2(128, 3),
            ResidualLayerV2(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

class TransformerResNextNetwork(nn.Module):
    """
    Feedforward Transformation Network - ResNeXt
    
        - No Tanh
        + ResNeXt Layer
    """
    def __init__(self):
        super(TransformerResNextNetwork, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

class TransformerResNextNetwork_Pruned(nn.Module):
    """
    Feedforward Transformation Network - ResNeXt
    
        - No Tanh
        + ResNeXt Layer
        + Pruned
        Reference: https://heartbeat.fritz.ai/creating-a-17kb-style-transfer-model-with-layer-pruning-and-quantization-864d7cc53693 
    """
    def __init__(self, alpha=1.0):
        super(TransformerResNextNetwork_Pruned, self).__init__()
        a = alpha
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, int(a*32), 9, 1),
            nn.ReLU(),
            ConvLayer(int(a*32), int(a*32), 3, 2),
            nn.ReLU(),
            ConvLayer(int(a*32), int(a*32), 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
            ResNextLayer(int(a*32), [int(a*16), int(a*16), int(a*32)], kernel_size=3),
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(int(a*32), int(a*32), 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(int(a*32), int(a*32), 3, 2, 1),
            nn.ReLU(),
            ConvLayer(int(a*32), 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

class TransformerNetworkDenseNet(nn.Module):
    """
    Feedforward Transformer Network using DenseNet Block instead of Residual Block
    """
    def __init__(self):
        super(TransformerNetworkDenseNet, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayerNB(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayerNB(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayerNB(64, 128, 3, 2),
            nn.ReLU()
        )
        self.DenseBlock = nn.Sequential(
            NormReluConv(128, 64, 1, 1),
            DenseLayerBottleNeck(64, 16),
            DenseLayerBottleNeck(80, 16),
            DenseLayerBottleNeck(96, 16),
            DenseLayerBottleNeck(112, 16)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.DenseBlock(x)
        out = self.DeconvBlock(x)
        return out

class TransformerNetworkUNetDenseNetResNet(nn.Module):
    """
    Feedforward Transformer Network using DenseNet Block instead of Residual Block
    """
    def __init__(self):
        super(TransformerNetworkUNetDenseNetResNet, self).__init__()
        self.C1 = ConvLayerNB(3, 32, 9, 1)
        self.RC1 = nn.ReLU()
        self.C2 = ConvLayerNB(32, 64, 3, 2)
        self.RC2 = nn.ReLU()
        self.C3 = ConvLayerNB(64, 128, 3, 2)
        self.RC3 = nn.ReLU()
        self.DenseBlock = nn.Sequential(
            NormReluConv(128, 64, 1, 1),
            DenseLayerBottleNeck(64, 16),
            DenseLayerBottleNeck(80, 16),
            DenseLayerBottleNeck(96, 16),
            DenseLayerBottleNeck(112, 16)
        )
        self.RD0 = nn.ReLU()
        self.D1 = UpsampleConvLayer(128, 64, 3, 1, 2)
        self.RD1 = nn.ReLU()
        self.D2 = UpsampleConvLayer(64, 32, 3, 1, 2)
        self.RD2 = nn.ReLU()
        self.D3 = ConvLayerNB(32, 3, 9, 1, norm="None")

    def forward(self, x):
        # Decoder
        x = self.RC1(self.C1(x))
        i1 = x
        x = self.RC2(self.C2(x))
        i2 = x
        x = self.RC3(self.C3(x))
        i3 = x
        
        # Dense Block
        x = self.DenseBlock(x)
        if (x.shape != i3.shape):
            sh = i3.shape
            x = x[:sh[0], :sh[1], :sh[2], :sh[3]] + i3
        else:
            x = x + i3
        x = self.RD0(x)

        # Encoder
        x = self.D1(x)
        if (x.shape != i2.shape):
            sh = i2.shape
            x = x[:sh[0], :sh[1], :sh[2], :sh[3]] + i2
        else:
            x = x + i2

        x = self.RD1(x)
        x = self.D2(x)
        if (x.shape != i1.shape):
            sh = i1.shape
            x = x[:sh[0], :sh[1], :sh[2], :sh[3]] + i1
        else:
            x = x + i1
        x = self.RD2(x)
        x = self.D3(x)
        
        return x

class DenseLayerBottleNeck(nn.Module):
    """
    NORM - RELU - CONV1 -> NORM - RELU - CONV3

    out_channels = Growth Rate
    """
    def __init__(self, in_channels, out_channels):
        super(DenseLayerBottleNeck, self).__init__()

        self.conv1 = NormLReluConv(in_channels, 4*out_channels, 1, 1)
        self.conv3 = NormLReluConv(4*out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.conv3(self.conv1(x))
        out = torch.cat((x, out), 1)
        return out

class ConvLayerNB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayerNB, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class ResidualLayerV2(nn.Module):
    """
    Identity Mappings in Deep Residual Networks
        
        Full pre-activation

    https://arxiv.org/abs/1603.05027
    """
    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayerV2, self).__init__()
        self.conv1 = NormReluConv(channels, channels, kernel_size, stride=1)
        self.conv2 = NormReluConv(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out

class ResNextLayer(nn.Module):
    """
    Aggregated Residual Transformations for Deep Neural Networks

        Equal to better performance with 10x less parameters

    https://arxiv.org/abs/1611.05431
    """
    def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
        super(ResNextLayer, self).__init__()
        ch1, ch2, ch3 = channels
        self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out = out + identity
        return out

class NormReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(NormReluConv, self).__init__()

        # Normalization Layers
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        # ReLU Layer
        self.relu_layer = nn.ReLU()

        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.relu_layer(x)
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x

class NormLReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(NormLReluConv, self).__init__()

        # Normalization Layers
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        # ReLU Layer
        self.relu_layer = nn.ReLU()

        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.relu_layer(x)
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out