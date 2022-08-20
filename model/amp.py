import torch
import torch.nn as nn

from model.gan import Encoder, Decoder

class AMP_GAN(nn.Module):
    def __init__(self, size, n_channels, stages, scale=2):
        super(AMP_GAN, self).__init__()
        self.size = size
        self.n_channels = n_channels
        self.stages = stages
        self.scale = scale

        self.encoder = Encoder(n_channels, scale=scale)
        self.decoder = Decoder(n_channels, scale=scale)
        self.measurement = nn.Parameter(torch.normal(0, 1 / (size // scale)), requires_grad=True)
        self.amp_stage = AMP_Stage(size, n_channels, scale=scale)

    def forward(self, x):
        assert x.shape[1] == self.size
        assert x.shape[2] == self.size

        y = self.measurement @ x
        out = torch.t(self.measurement) @ y
        for i in range(self.stages):
            out = self.amp_stage(out, y, self.measurement)
        out = self.decoder(out)
        return out


class AMP_Stage(nn.Module):
    def __init__(self, size, n_channels, scale=2, lamda=1):
        super(AMP_Stage, self).__init__()
        self.size = size
        self.n_channels = n_channels
        self.scale = scale
        self.lamda = lamda
        self.alpha = nn.Parameter(torch.ones(1))

    def _eta(self, x):
        # soft thresholding
        return torch.sign(x) * torch.maximum(torch.abs_(x) - self.lamda * self.alpha.unsqueeze(1), 0)

    def _d_eta(self, x):
        # derivative of soft thresholding
        return torch.where(torch.abs_(x) < self.lamda, torch.zeros_like(x), torch.ones_like(x))

    def forward(self, x, y, h, measurement):
        z = y - measurement @ x + h
        h_t = 
        return self._eta(self.alpha.unsqueeze(1) * torch.t(measurement) @ z + x), h_t