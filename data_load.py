import torchvision.datasets as datasets
import os

class ImageLoader(object):
    def __init__(self, path):
        self._len = 0
        self.image = []
        files = os.listdir(path)
        for file in files:
            if file[0] != '.' and file.split('.')[-1] in ['jpg', 'png']:
                self._len += 1
                self.image.append(path  + "/" + file)
        # print(self.image)

    def get_image_filename(self, index):
        return self.image[index]