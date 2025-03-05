import torch
import torch.nn as nn


class Direct(nn.Module):
    def __init__(self, in_channel, out_channel, h, w):
        super(Direct, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.w = w
        self.h = h
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(h, w))
        self.act = nn.Softmax(dim=-3)

    def forward(self, x):
        x = self.conv(x).reshape()
        x = self.act(x)
        return x


class Direct_TF(nn.Module):
    def __init__(self, in_channel, out_channel, h, w, l):
        super(Direct_TF, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.w = w
        self.h = h
        self.l = l
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(h, w))
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.l, self.out_channel // self.l)
        x = self.act(x)
        return x
