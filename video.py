import torch
import utils
import transformer
import cv2
import os
from stylize import stylize_folder_single, stylize_folder

VIDEO_NAME = "dance.mp4"
FRAME_SAVE_PATH = "frames/"
FRAME_CONTENT_FOLDER = "content_folder/"
FRAME_BASE_FILE_NAME = "frame"
FRAME_BASE_FILE_TYPE = ".jpg"
STYLE_FRAME_SAVE_PATH = "style_frames/"
STYLE_VIDEO_NAME = "helloworld.mp4"
STYLE_PATH = "transforms/mosaic_aggressive.pth"
BATCH_SIZE = 20

import time

def video_transfer(video_path, style_path):
    print("OpenCV {}".format(cv2.__version__))
    starttime = time.time()
    # Extract video info
    H, W, fps = getInfo(video_path)
    print("Height: {} Width: {} FPS: {}".format(H, W, fps))

    # Extract all frames
    print("Extracting video frames")
    getFrames(video_path)
    
    # Stylize a directory
    print("Performing style transfer on frames")
    #stylize_folder_single(style_path, FRAME_SAVE_PATH, STYLE_FRAME_SAVE_PATH)
    stylize_folder(style_path, FRAME_SAVE_PATH, STYLE_FRAME_SAVE_PATH, batch_size=BATCH_SIZE)

    # Combine all frames
    print("Combining style frames into one video")
    makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, fps, int(H), int(W))
    print("Elapsed Time: {}".format(time.time()-starttime))
    tor

def getInfo(video_path):
    """
    Extracts the height, width,
    and fps of a video
    """
    vidcap = cv2.VideoCapture(video_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vidcap.get(cv2.CAP_PROP_FPS)
    return height, width, fps

def getFrames(video_path):
    """
    Extracts the frames of a video
    and saves in specified path
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite("{}{}{}{}".format(FRAME_SAVE_PATH+FRAME_CONTENT_FOLDER, FRAME_BASE_FILE_NAME, count, FRAME_BASE_FILE_TYPE), image)
        success, image = vidcap.read()
        count+=1
    print("Done extracting all frames")
    
def makeVideo(frames_path, save_name, fps, height, width):    
    # Extract image paths. Natural sorting of directory list. Python does not have a native support for natural sorting :(
    base_name_len = len(FRAME_BASE_FILE_NAME)
    filetype_len = len(FRAME_BASE_FILE_TYPE)
    images = [img for img in sorted(os.listdir(frames_path), key=lambda x : int(x[base_name_len:-filetype_len])) if img.endswith(".jpg")]
    
    # Define the codec and create VideoWrite object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    vout = cv2.VideoWriter(save_name, fourcc, fps, (width,height))

    # Write the video
    for image_name in images:
        vout.write(cv2.imread(os.path.join(frames_path, image_name)))

    print("Done writing video")

video_transfer(VIDEO_NAME, STYLE_PATH)