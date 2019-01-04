import torch
import utils
import transformer

STYLE_TRANSFORM_PATH = "transforms/tokyo_ghoul3.pth"
CONTENT_IMAGE_PATH = "images/sungha-jung.jpg"

def stylize():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH))
    net = net.to(device)

    # Load image
    content_image = utils.load_image(CONTENT_IMAGE_PATH)
    print("1")
    content_tensor = utils.itot(content_image).to(device)
    print("2")
    generated_tensor = net(content_tensor)
    print("3")
    generated_image = utils.ttoi(generated_tensor.clone().detach())
    print("4")

    utils.show(generated_image)
    utils.saveimg(generated_image, "rusty_tg.png")

stylize()