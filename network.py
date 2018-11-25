import torch.nn as nn
from torch.nn.modules import Sequential


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self._scale_factor = scale_factor

    def forward(self, X):
        return nn.functional.interpolate(X, scale_factor=self._scale_factor)


class StyleBankNet(nn.Module):
    def __init__(self):
        super(StyleBankNet , self).__init__()
        self.encoder_net = Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stylebank_net = Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder_net = Sequential(
            Interpolate(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            Interpolate(2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, bias=False)
        )

    def forward(self, X, style_id=None):
        out = self.encoder_net(X)
        out = self.stylebank_net(out)
        return self.decoder_net(out)
