import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from network import StyleBankNet
from data_load import MyDataSet
from loss_function import Loss

if __name__ == '__main__':
    print('train begin:')
    CONTENT_WEIGHT = 1
    STYLE_WEIGHT = 100
    LOG_INTERVAL = 200
    REGULARIZATION = 1e-7
    EPOCH = 100
    T = 5
    LAMBDA = 1
    LR = 1e-2
    mse_loss = torch.nn.MSELoss()
    style_loss_features = Loss(Variable(style_img_tensor, volatile=True))
    gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features]

    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    train_data = MyDataSet(path="./content", transform=transform)
    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    optimizer = Adam(StyleBankNet.parameters(), LR, )
    StyleBankNet.train()

    for epoch in range(EPOCH):
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_reg_loss = 0.
        agg_identity_loss = 0.
        count = 0
        t = 0

        for batch_id, (x, _) in enumerate(data_loader):
            optimizer.zero_grad()

            n_batch = len(x)
            count += n_batch
            x = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda()
            y = StyleBankNet(X=x)

            xc = Variable(x.data, volatile=True)
            features_y = Loss(y)
            features_xc = Loss(xc)

            # content loss at relu4_2
            f_xc_c = Variable(features_xc[3].data, requires_grad=False)
            content_loss = CONTENT_WEIGHT * mse_loss(features_y[3], f_xc_c)
            reg_loss = REGULARIZATION * (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = gram_style[m]
                gram_y = gram_matrix(features_y[m])
                style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s.expand_as(gram_y))

            total_loss = content_loss + style_loss + reg_loss
            total_loss.backward()
            optimizer.step()