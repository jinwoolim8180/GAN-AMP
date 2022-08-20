import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Encoder(nn.Module):
    def __init__(self, n_channels, scale=2):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            weight_norm(nn.Conv2d(3, n_channels // 2, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(n_channels // 2, n_channels // 2, kernel_size=scale, stride=scale)),
            weight_norm(nn.Conv2d(n_channels // 2, n_channels, kernel_size=1))
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, n_channels, scale=2):
        super(Decoder, self).__init__()
        self.conv_trans = nn.Sequential(
            weight_norm(nn.ConvTranspose2d(n_channels, n_channels // 2, kernel_size=1)),
            weight_norm(nn.ConvTranspose2d(n_channels // 2, n_channels // 2, kernel_size=scale, stride=scale)),
            weight_norm(nn.ConvTranspose2d(n_channels // 2, 3, kernel_size=3, padding=1)),
            nn.Tanh()
        )

    def forward(self, z):
        return self.conv_trans(z)


class Discriminator(nn.Module):
    def __init__(self, n_channels, size, num_pool=4):
        super(Discriminator, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_pool - 1):
            chan = n_channels // (2 ** (num_pool - i - 2))
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(3, chan, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(chan),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
        self.linear = nn.Linear(size ** 2 // (2 ** (num_pool - 1)), 1)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return self.linear(out)