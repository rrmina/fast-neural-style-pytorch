import cv2
import transformer
import torch
import utils

STYLE_TRANSFORM_PATH = "transforms/mosaic.pth"
PRESERVE_COLOR = False
WIDTH = 1280
HEIGHT = 720

def webcam(style_transform_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    print("Loading Transformer Network")
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_transform_path))
    net = net.to(device)
    print("Done Loading Transformer Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        while True:
            # Get webcam input
            ret_val, img = cam.read()

            # Mirror 
            img = cv2.flip(img, 1)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Generate image
            content_tensor = utils.itot(img).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(img, generated_image)

            generated_image = generated_image / 255

            # Show webcam
            cv2.imshow('Demo webcam', generated_image)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
            
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

webcam(STYLE_TRANSFORM_PATH, WIDTH, HEIGHT)
