import torch
from torch.autograd import Variable
from collections import namedtuple
from torch.nn.modules import Sequential

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_2", "relu4_2"])
MSE = torch.nn.MSELoss()

def gram_matrix(data_in):
    batch, channels, height, width = data_in.size()
    data = data_in.view(batch, channels, height * width)
    # [batch, channels, height * width] * [batch, height * width, channels]
    gram = torch.bmm(data, data.transpose(1, 2)) / (channels * height * width)
    return gram

def tv_loss(data_in):
    batch, channels, height, width = data_in.size()
    height_size = channels * (height - 1) * width
    width_size = channels * height  * (width - 1)
    tv_height =MSE(data_in[:, :, 1:, :], data_in[:, :, :-1, :])
    tv_width = MSE(data_in[:, :, :, 1:], data_in[:, :, :, :-1])
    return (tv_height / height_size + tv_width / width_size ) / batch

class StyleBnakLoss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(StyleBnakLoss, self).__init__()
        self.content_weight = 1
        self.style_weight = 100
        self.tv_weight = 1
        self.vgg_layer_content_used = {
            '20': "relu4_2"
        }
        self.vgg_layer_style_used = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '20': "relu4_2"
        }
        self.content_loss = Sequential()
        self.style_loss = Sequential()
        for name, module in enumerate(vgg_model):  # name: '3', module: the net
            if name in self.vgg_layer_content_used:
                self.content_loss.add_module(self.vgg_layer_content_used[name], module)
            if name in self.vgg_layer_style_used:
                self.style_loss.add_module(self.vgg_layer_style_used[name], module)

    def forward(self, O, I, S):
        # print('Output size:', O.size())
        batch, channels, height, width = O.size()
        for i in range(batch):
            content_loss = MSE(self.content_loss(O[i].view(1,channels, height, width)), (self.content_loss(I)))
        style_loss =  MSE(gram_matrix(self.style_loss(O)), gram_matrix((self.style_loss(S))))
        total_variation_loss = tv_loss(O)

        loss = self.content_weight * content_loss + self.style_weight * style_loss + self.tv_weight * total_variation_loss
        return loss
