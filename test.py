import torch
from torch.autograd import Variable
from network import StyleBankNet
from dot import  make_dot

def test_net():
    x = Variable(torch.randn(1,3,20,20))#change 12 to the channel number of network input
    net = StyleBankNet()
    print(net)
    # y = net(x)
    # g = make_dot(y)
    # g.view()


from data_load import ImageLoader, MyDataSet
from utils import *

def test_image_load():
    image_list = ImageLoader("./content")
    filename = image_list.get_image_filename(0)
    # show_file(filename)
    img =  load_image(filename)
    transform = transforms.Compose([
        transforms.Resize(224),  # 缩放图片，保持长宽比不变，最短边的长为224像素,
        transforms.CenterCrop(224),  # 从中间切出 224*224的图片
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    img = transform(img)
    img = torchvision.utils.make_grid(img)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.show()


import torchvision
from torchvision import transforms
def test_DataLoader():
    # 加上transforms
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    dataset = torchvision.datasets.ImageFolder('F:/workplace/python/DL/Olio', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

    dataiter = iter(dataloader)
    imgs, labels = next(dataiter)
    print(dataset.class_to_idx)
    print(imgs.size(), labels)
    for i in range(len(dataset.imgs)):
        print(dataset.imgs[i])

from torch.utils.data import DataLoader
def show_batch(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

def test_MyDataLoader():
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    train_data = MyDataSet(path="./content", transform = transform)
    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    print(len(data_loader))
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if (i < 4):
            print(i, batch_x.size(), batch_y.size())
            show_batch(batch_x)
            plt.axis('off')
            plt.show()

from torchvision import models
def model_vgg16_test():
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    print(vgg16.features)
    # for k, v in pretrained_dict.items():
    #     print(k, v.size())

if __name__ == '__main__':
    # test_net()
    # test_image_load()
    # test_DataLoader()
    # test_MyDataLoader()
    model_vgg16_test()
    print('ok')