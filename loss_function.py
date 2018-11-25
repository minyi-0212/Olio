import torch
from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_2", "relu4_2"])
class Loss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(Loss, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '20': "relu4_2"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)