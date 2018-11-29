import torch
from torch.autograd import Variable
from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_2", "relu4_2"])
MSE = torch.nn.MSELoss()
class StyleBnakLoss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(StyleBnakLoss, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '20': "relu4_2"
        }

    def forward(self, x):
        output = {}
        style = x
        # print(self.vgg_layers._modules['20'])
        content = self.vgg_layers._modules['20'](x)
        for name, module in self.vgg_layers._modules.items():  #name: '3', module: the net
            style = module(style)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = style
        # return LossOutput(**output)
        # return  torch.nn.MSELoss(out, x)
        # out = Variable(style.data, requires_grad=False)
        loss = MSE(content, x)
        return loss