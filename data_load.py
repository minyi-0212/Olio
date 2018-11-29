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

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataSet(Dataset):
    def __init__(self, path, transform=None, target_transform=None, loader=default_loader, size = None):
        images = []
        id = 0
        files = os.listdir(path)
        if size is not None:
            for file in files:
                if id >= size:
                    break
                if file[0] != '.' and file.split('.')[-1] in ['jpg', 'png']:
                    images.append([path + "/" + file, id])
                    id += 1
        else:
            for file in files:
                if file[0] != '.' and file.split('.')[-1] in ['jpg', 'png']:
                    images.append([path + "/" + file, id])
                    id += 1


        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        file, label = self.imgs[index]
        img = self.loader(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)