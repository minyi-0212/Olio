import pylab
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def show_file(filename, size=None, scale=None):
    p = plt.subplot()
    p.set_title("image file: %5s" % filename)
    plt.imshow(load_image(filename, size, scale))
    pylab.show()

def show_image(image, size=None, scale=None):
    p = plt.subplot()
    plt.imshow(image)
    pylab.show()

def show_imgs(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.show()