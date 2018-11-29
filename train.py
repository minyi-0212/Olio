from utils import *
from config import config
from data_load import MyDataSet
from network import StyleBankNet
from loss_function import StyleBnakLoss

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models

if __name__ == '__main__':
    cfg = config()
    print('content file : {}'.format(cfg['content_filepath']))

    # load data
    transform = transforms.Compose([
        transforms.CenterCrop(50),
        transforms.ToTensor()
    ])
    content_data = MyDataSet(path=cfg['content_filepath'], transform=transform)
    content_data_loader = DataLoader(content_data, batch_size=1, shuffle=True)
    style_data = MyDataSet(path=cfg['style_filepath'], transform=transform, size=cfg['style_size'])
    style_data_loader = DataLoader(style_data, batch_size=cfg['style_size'])
    style_imgs, style_labels = next(iter(style_data_loader))

    if cfg['debug'] != 0:
        print(len(content_data_loader))
        for index, (image, label) in enumerate(content_data_loader):
            if (index < 2):
                print(index, image.size(), label.size())
                grid = utils.make_grid(image)
                plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.title('show data')
                plt.axis('off')
                plt.show()

    # init
    style_bank_net = StyleBankNet(cfg['style_size'])
    vgg16 = models.vgg16(pretrained=True)
    style_bank_loss = StyleBnakLoss(vgg16)
    style_bank_net.train()  # 把模型的状态设置为训练状态，主要针对Dropout层
    # optimizer = torch.optim.SGD(style_bank_net.parameters())
    optimizer = torch.optim.SGD(style_bank_net.parameters(), lr = cfg['lr'], momentum = cfg['momentum'])
    cuda_is_available = torch.cuda.is_available() & cfg['GPU']
    if cuda_is_available:
        print('cuda...')
        style_bank_net.cuda()
        style_bank_loss.cuda()

    # train
    for epoch in range(cfg['epochs']):
        for index, (image, label) in enumerate(content_data_loader):
            if cuda_is_available:
                print('cuda...')
                data = image.cuda()
            #forward
            # output = style_bank_net(image)  # 对data做前向过程，得到输出
            output = style_bank_net(image, style_imgs, style_labels)  # 对data做前向过程，得到输出
            loss = style_bank_loss(output)  # 计算output和target之间的损失
            print('train loss:', loss)
            # loss.backward()  # 反向过程，计算损失关于各参数的梯度
            # optimizer.step()  # 利用计算得到的梯度对参数进行更新

