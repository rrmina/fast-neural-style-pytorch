import torch
import utils
import transformer
import os

STYLE_TRANSFORM_PATH = "transforms/mosaic_dark2.pth"
#CONTENT_IMAGE_PATH = "images/rusty.jpg"

def stylize():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH))
    net = net.to(device)

    with torch.no_grad():
        while(1):
            torch.cuda.empty_cache()
            print("Stylize Image~ Press Ctrl+C and Enter to close the program")
            content_image_path = input("Enter the image path: ")
            content_image = utils.load_image(content_image_path)
            content_tensor = utils.itot(content_image).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            utils.show(generated_image)

def stylize_folder_single(style_path, content_folder, save_folder):
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    # Stylize every frame
    images = [img for img in os.listdir(content_folder) if img.endswith(".jpg")]
    with torch.no_grad():
        for image_name in images:
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Load content image
            content_image = utils.load_image(content_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())

            # Save image
            utils.saveimg(generated_image, save_folder + image_name)

#stylize()