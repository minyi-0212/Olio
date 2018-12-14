from utils import *
from config import config
from data_load import MyDataSet
from network import StyleBankNet
from loss_function import StyleBnakLoss, MSE

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import cv2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.xavier(m.weight.data)
        torch.nn.init.xavier(m.bias.data)

if __name__ == '__main__':
    cfg = config()
    print('content file : {}'.format(cfg['content_filepath']))

    # load data
    img_size = cfg['img_size']
    transform = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    content_data = MyDataSet(path=cfg['content_filepath'], transform=transform)
    content_data_loader = DataLoader(content_data, batch_size=cfg['content_batch_size'], shuffle=False)
    style_data = MyDataSet(path=cfg['style_filepath'], transform=transform, size=cfg['style_size'])
    style_data_loader = DataLoader(style_data, batch_size=cfg['style_size'])
    style_imgs, style_labels = next(iter(style_data_loader))

    if cfg['debug'] != 0:
        print(len(content_data_loader))
        for index, (image, label) in enumerate(content_data_loader):
            if index < 2:
                print(index, image.size(), label.size())
                grid = utils.make_grid(image)
                plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.title('show data')
                plt.axis('off')
                plt.show()

    # init
    style_bank_net = StyleBankNet(cfg['style_size'])
    # style_bank_net.apply(weights_init)
    style_bank_net.train()

    content_weight = cfg['content_weight']
    style_weight = cfg['style_weight']
    tv_weight = cfg['tv_weight']

    vgg16 = models.vgg16(pretrained=True).features
    style_bank_loss = StyleBnakLoss(vgg16)
    optimizer = torch.optim.Adam(style_bank_net.parameters(), lr=cfg['lr'])
    optimizer_ed = torch.optim.Adam(style_bank_net.parameters(), lr=cfg['lr'])

    grid = torchvision.utils.make_grid(style_imgs.view(-1, 3, img_size, img_size), nrow=cfg['style_size'])
    plt.imsave('output_image/style.png', grid.numpy().transpose((1, 2, 0)))

    content_image, content_label = next(iter(content_data_loader))
    grid = torchvision.utils.make_grid(content_image.view(-1, 3, img_size, img_size),  nrow=cfg['style_size'])
    plt.imsave('output_image/content.png', grid.numpy().transpose((1, 2, 0)))

    cuda_is_available = torch.cuda.is_available() & cfg['GPU']
    if cuda_is_available:
        print('cuda...')
        style_bank_net.cuda()
        style_bank_loss.cuda()
        content_image = content_image.cuda()
        style_imgs = style_imgs.cuda()
        style_labels = style_labels.cuda()

    # train
    t = Timer()
    t.tic()
    for epoch in range(cfg['begin_epochs'], cfg['epochs']):
        if epoch == 0 or (epoch+1) % 200 == 0:
            print('*', end=' ')
        # stylizing branch
        for branch_epoch in range(cfg['T']):
            # forward
            optimizer.zero_grad()
            output = style_bank_net(content_image, style_imgs, style_labels)
            content_loss, style_loss, tv_loss = style_bank_loss(output, content_image, style_imgs)
            loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
            # output = style_bank_net(content_image)
            # loss = MSE(output, content_image)
            # print('{}/{} iter, index {}   train loss: {}'.format(epoch, cfg['epochs']+1, index, loss))
            loss.backward()
            optimizer.step()

            if epoch == 0 or (epoch+1) % cfg['LOG_INTERVAL'] == 0:
                print('{}/{} iter, style_branch {}, train loss: {}, {}, {}, {}'.format(epoch+1, cfg['epochs'], branch_epoch,content_weight * content_loss, style_weight * style_loss, tv_weight * tv_loss, loss))
                # print('content loss: {}'.format(loss))
                grid = torchvision.utils.make_grid(output.data.view(-1, 3, img_size, img_size),  nrow=cfg['style_size']).cpu().numpy().transpose(
                    (1, 2, 0))
                plt.imsave('output_image/iter{}_{}_style_branch{}'.format(epoch+1, cfg['epochs'], branch_epoch), grid)
                # print(grid.size())
                # plt.imshow(grid)
                # plt.title('{}/{} iter, index {}'.format(epoch+1, cfg['epochs'], index))
                # plt.show()
        # enoder-decoder branch
        optimizer_ed.zero_grad()
        output = style_bank_net(content_image)
        # print(content_image.size())
        # print(output.size())
        loss = MSE(output, content_image)
        # print(loss)
        loss.backward()
        optimizer_ed.step()
        if epoch == 0 or (epoch+1) % cfg['LOG_INTERVAL'] == 0:
            diff_time = t.toc()
            print('time {}, ED content loss: {}'.format(diff_time, loss))
            grid = torchvision.utils.make_grid(output.data.view(-1, 3, img_size, img_size), nrow=5).cpu().numpy().transpose(
                (1, 2, 0))
            plt.imsave('output_image/iter{}_{}_ED'.format(epoch+1, cfg['epochs']), grid)

            # save whole model (including stylebank)
            torch.save(style_bank_net.state_dict(), cfg['ouput_path']+'/bank.pth')
            # save seperate part
            torch.save(style_bank_net.encoder_net.state_dict(), cfg['ouput_path']+'/encode.pth')
            torch.save(style_bank_net.decoder_net.state_dict(), cfg['ouput_path']+'/decode.pth')
            for i in range(len(style_imgs)):
                torch.save(style_bank_net.stylebank_net[i].state_dict(), cfg['ouput_path']+'/style{}.pth'.format(i))
            t.tic()

        # if (epoch + 1) % cfg['update_lr'] == 0:
        #     lr = cfg['lr'] * (0.8 ** ((epoch + 1)/cfg['update_lr']))
        #     print('learning rate: {}'.format(lr))
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     for param_group in optimizer_ed.param_groups:
        #         param_group['lr'] = lr
