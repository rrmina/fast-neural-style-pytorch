# fast-neural-style: Fast Style Transfer in Pytorch! :art:

An implementation of **fast-neural-style** in PyTorch! Style Transfer learns the aesthetic style of a `style image`, usually an art work, and applies it on another `content image`. This repository contains codes the can be used for:
1. fast `image-to-image` aesthetic style transfer, 
2. `image-to-video` aesthetic style transfer, and for
3. training `style-learning` transformation network

This implemention follows the style transfer approach outlined in [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) paper by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li*, along with the [supplementary paper detailing the exact model architecture](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf) of the mentioned paper. The idea is to train a **`separate feed-forward neural network (called Transformation Network) to transform/stylize`** an image and use backpropagation to learn its parameters, instead of directly manipulating the pixels of the generated image as discussed in [A Neural Algorithm of Artistic Style aka **neural-style**](https://arxiv.org/abs/1508.06576) paper by *Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge*. The use of feed-forward transformation network allows for fast stylization of images, around 1000x faster than neural style.

This implementation made some **modifications** in Johnson et. al.'s proposed architecture, particularly:
1. The use of **`reflection padding in every Convolutional Layer`**, instead of big single reflection padding before the first convolution layer
2. **`Ditching of Tanh in the output layer`**. The generated image are the raw outputs of the convolutional layer. While the Tanh model produces visually pleasing results, the model fails to transfer the vibrant and loud colors of the style image (i.e. generated images are usually darker). This however makes for a good **`retro style effect`**)
3. Use of **`Instance Normalization`**, instead of Batch Normalization after Convolutional and Deconvolutional layers, as discussed in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) paper by *Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky*.

The [original caffe pretrained weights of VGG16](https://github.com/jcjohnson/pytorch-vgg) were used for this implementation, instead of the pretrained VGG16's in PyTorch's model zoo.

# Image Stylization

# Video Stylization

# Webcam Demo


## Requirements
Most of the codes here assume that the user have access to CUDA capable GPU, at least a GTX 1050 ti or a GTX 1060
### Data Files
* [Pre-trained VGG16 network weights](https://github.com/jcjohnson/pytorch-vgg) - put it in `models/` directory
* [MS-COCO Train Images (2014)](http://cocodataset.org/#download) - 13GB - put `train2014` directory in `dataset/` directory
* [torchvision](https://pytorch.org/) - `torchvision.models` contains the VGG16 and VGG19 model skeleton

### Dependecies
* [PyTorch](https://pytorch.org/)
* [opencv2](https://matplotlib.org/users/installing.html)
* [NumPy](https://www.scipy.org/install.html)
* [FFmpeg](https://www.ffmpeg.org/) (Optional) - Insllation [Instruction here](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg)

# Usage
All arguments, parameters and options are **`hardcoded`** inside these 5 python files.  
## Training Style Transformation Network
**`train.py`**: trains the transformation network that learns the style of the `style image`
```
python train.py
```

**Options**
* `TRAIN_IMAGE_SIZE`: sets the dimension (height and weight) of training images. Bigger GPU memory is needed to train with larger images. Default is `256`px.
* `DATASET_PATH`: folder containing the MS-COCO `train2014` images. Default is `"dataset"` 
* `NUM_EPOCHS`: Number of epochs of training pass. Default is `1` with 40,000 training images
* `STYLE_IMAGE_PATH`: path of the style image
* `BATCH_SIZE`: training batch size. Default is 4 
* `CONTENT_WEIGHT`: Multiplier weight of the loss between content representations and the generated image. Default is `2e-6`
* `STYLE_WEIGHT`: Multiplier weight of the loss between style representations and the generated image. Default is `5e0`
* `ADAM_LR`: learning rate of the adam optimizer. Default is `0.001`
* `SAVE_MODEL_PATH`: path of pretrained-model weights and transformation network checkpoint files. Default is `"models/"`
* `SAVE_IMAGE_PATH`: save path of sample tranformed training images. Default is `"images/out/"`
* `SAVE_MODEL_EVERY`: Frequency of saving of checkpoint and sample transformed images. 1 iteration is defined as 1 batch pass. Default is `500` with batch size of `4`, that is 2,000 images
* `SEED`: Random seed to keep the training variations as little as possible

**`transformer.py`**: contains the architecture definition of the tranformation network. It includes 2 models, `TransformerNetwork()` and `TransformerNetworkTanh()`. `TransformerNetwork` doesn't have an extra output layer, while `TransformerNetworkTanh`, as the name implies, has for its output, a Tanh layer and a default `output multiplier of 150`. `TransformerNetwork` faithfully copies the style and colorization of the style image, while Tanh model produces images with darker color; which brings a **`retro style effect`**.
**Options** 
* `norm`: sets the normalization layer to either Instance Normalization `"instance"` or Batch Normalization `"batch"`. Default is `"instance"`
* `tanh_multiplier`: output multiplier of the Tanh model. The bigger the number, the bright the image. Default is `150`

## Stylizing Images
**`stylize.py`**: Loads a pre-trained transformer network weight and applies style (1) to a content image or (2) to the images inside a folder
```
python stylize.py
```
**Options**
* `STYLE_TRANSFORM_PATH`: path of the pre-trained weights of the the transformation network. Sample pre-trained weights are availabe in `transforms` folder, including their implementation parameters.

## Stylizing Videos
**`video.py`**: Extracts all frames of a video, apply fast style transfer on each frames, and combine the styled frames into an output video. The output video doesn't retain the original audio. Optionally, you may use FFmpeg to merge the output video and the original video's audio.
```
python video.py
```
**Options**
* `VIDEO_NAME`: path of the original video
* `FRAME_SAVE_PATH`: parent folder of the save path of the extracted original video frames. Default is `"frames/"`
* `FRAME_CONTENT_FOLDER`: folder of the save path of the extracted original video frames. Default is `"content_folder/"`
* `FRAME_BASE_FILE_NAME`: base file name of the extracted original video frames.  Default is  `"frame"`
* `FRAME_BASE_FILE_TYPE`: save image file time ".jpg"
* `STYLE_FRAME_SAVE_PATH`: path of the styled frames. Default is `"style_frames/"`
* `STYLE_VIDEO_NAME`: name(or save path) of the output styled video. Default is `"helloworld.mp4"`
* `STYLE_PATH`: pretrained weight of the style of the transformation network to use for video style transfer. Default is `"transforms/mosaic_aggressive.pth"`
* `BATCH_SIZE`: batch size of stylization of extracted original video frames. A 1080ti 11GB can handle a batch size of 20 for 720p videos, and 80 for a 480p videos. Dafult is `1`
* `USE_FFMPEG`(Optional): Set to `True` if you want to use FFmpeg in extracting the original video's audio and encoding the styled video with the original audio.

## Stylizing Webcam
**`webcam.py`**: Captures and saves webcam output image, perform style transfer, and again saves a styled image. Reads the styled image and show in window. 
```
python webcam.py
```
**Options**
* `STYLE_TRANSFORM_PATH`: pretrained weight of the style of the transformation network to use for video style transfer. Default is `"transforms/mosaic_aggressive.pth"`
* `WIDTH`: width of the webcam output window. Default is `1280`
* `HEIGHT`: height of the webcam output window. Default is `720`

## Files and Folder Structure
```
master_folder
 ~ dataset 
    ~ train2014
        coco*.jpg
        ...
 ~ frames
    ~ content_folder
        frame*.jpg
        ...
 ~ images
    ~ out
        *.jpg
      *.jpg
 ~ models
    *.pth
 ~ style_frames
    frames*.jpg
 ~ transforms
    *.pth
 *.py
```

## Todo!
* FFmpeg support for encoding videos with video style transfer
* Color-preserving Real-time Style Transfer
* ~~Webcam demo of fast-neural-style~~
* Web-app deployment of fast-neural-style (ONNX)

## Attribution
This implementation borrowed some implementation details from:
* Justin Johnson's [fast-neural-style in Torch](https://github.com/jcjohnson/fast-neural-style), and 
* the PyTorch Team's [PyTorch Examples: fast-neural-style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
* This repository also borrows some markdown formatting, as well as license description from Logan Engstrom's [fast-style-transfer in Tensorflow](https://github.com/lengstrom/fast-style-transfer)

## Related Work
* [Neural Style in PyTorch](https://github.com/iamRusty/neural-style-pytorch) - PyTorch implementation of the original [A Neural Algorithm of Artistic Style aka **neural-style**](https://arxiv.org/abs/1508.06576) paper by Gatys et. al.

## License

Copyright (c) 2018 Rusty Mina. For commercial use or any purpose that is not academic or personal, please contact me at (email: rustymina at gmail dot com). Free for academic or research use, as long as proper attribution is given and this copyright notice is retained.