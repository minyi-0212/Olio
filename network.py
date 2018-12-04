import torch
import torch.nn as nn
from torch.nn.modules import Sequential


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self._scale_factor = scale_factor

    def forward(self, X):
        return nn.functional.interpolate(X, scale_factor=self._scale_factor)


class StyleBankNet(nn.Module):
    def __init__(self, style_cnt):
        super(StyleBankNet , self).__init__()
        self.encoder_net = Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stylebank_net = nn.ModuleList([
            Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
            ) for i in range(style_cnt)])
        print('style_bank_cnt:{}'.format(style_cnt))

        self.decoder_net = Sequential(
            # Interpolate(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # Interpolate(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, bias=False)
        )

    def forward(self, content_in, style_data = None, style_label = None):
        if style_data is None:
            data_out = self.decoder_net(self.encoder_net(content_in))
            # print(data_out.size())
            return data_out
        else:
            data = []
            # print('content size {}'.format(len(content_in)))
            data.append(content_in)
            # print('style size {}'.format(len(style_data)))
            data.append(style_data)
            feature = torch.cat(data, dim=0)
            # print(feature.size())
            feature = self.encoder_net(feature)  # 128*H*W
            data_out = []
            content_size = len(content_in)
            for content_idx in range(content_size):
                for style_index in range(len(style_data)):
                    style_out = self.stylebank_net[style_label[style_index]](feature[content_size + style_index].view(1, *feature[style_index+1].shape))  # 128*H*W
                    data_out.append(style_out*feature[0])
            data_out = torch.cat(data_out, dim=0)
            data_out = self.decoder_net(data_out)
            # print(data_out.size())
            return data_out
