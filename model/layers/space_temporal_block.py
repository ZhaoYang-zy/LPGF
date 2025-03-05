import math

import torch
import torch.nn as nn
from model.layers.patch_wise_variable_aggregation import AttentionBlock



class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * 2 ** 2, out_channels)

    def forward(self, x):
        b, h, w, c = x.shape

        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = torch.reshape(x, (b, new_h, self.downscaling_factor, new_w, self.downscaling_factor, c))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = torch.reshape(x, (b, new_h, new_w, c * (self.downscaling_factor ** 2)))
        x = self.linear(x)

        return x


class PatchParting(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor):
        super().__init__()
        self.up_factor = up_factor
        self.linear = nn.Linear(in_channels, out_channels * (self.up_factor ** 2))

    def forward(self, x):

        x = self.linear(x)
        b, h, w, c = x.shape

        new_h, new_w = h * self.up_factor, w * self.up_factor
        x = torch.reshape(x, (b, h, w, c // (self.up_factor ** 2), self.up_factor, self.up_factor))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = torch.reshape(x, (b, new_h, new_w, c // (self.up_factor ** 2)))

        return x


class SWAttention(nn.Module):
    def __init__(self, stride, input_dim, hidden_dim, num_head, mask, nwin, win_patch, at_drop=0.0, pj_drop=0.0, variables = 16):
        super(SWAttention, self).__init__()
        self.B = nn.Parameter(torch.randn((variables, variables)))
        self.stride = stride
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.mask = mask
        self.nwin = nwin
        self.win_patch = win_patch
        self.sdk = torch.tensor(math.sqrt(hidden_dim))
        self.wq = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wk = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wv = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.proj = nn.Linear(in_features=hidden_dim * num_head, out_features=hidden_dim)
        self.sfm = torch.nn.Softmax(dim=-1)
        self.drop1 = nn.Dropout(at_drop)
        self.drop2 = nn.Dropout(pj_drop)

    def forward(self, x):
        # shift
        x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch,
                      self.input_dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.nwin[0] * self.win_patch,
                                                                        self.nwin[1] * self.win_patch, self.input_dim)
        x = torch.roll(x, (-self.stride, -self.stride), dims=(1, 2)).reshape(-1,
                                                                             self.nwin[0], self.win_patch, self.nwin[1],
                                                                             self.win_patch,
                                                                             self.input_dim).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.win_patch ** 2, self.input_dim)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)

        atten = q @ torch.transpose(k, -2, -1)
        atten = self.sfm(atten / self.sdk + self.B)
        atten = atten.reshape(-1, self.nwin[0], self.nwin[1], self.num_head, self.win_patch ** 2, self.win_patch ** 2)

        msk_atten = torch.zeros(atten.shape).cuda()
        msk_atten[:, 0:self.nwin[0] - 1, 0:self.nwin[1] - 1, :, :, :] = atten[:, 0:self.nwin[0] - 1, 0:self.nwin[1] - 1,
                                                                        :, :, :]
        msk_atten[:, -1, 0:self.nwin[1] - 1, :, :, :] = atten[:, -1, 0:self.nwin[1] - 1, :, :, :] * self.mask[0].cuda()
        msk_atten[:, 0:self.nwin[0] - 1, -1, :, :, :] = atten[:, 0:self.nwin[0] - 1, -1, :, :, :] * self.mask[1].cuda()
        msk_atten[:, -1, -1, :, :, :] = atten[:, -1, -1, :, :, :] * self.mask[2].cuda()
        msk_atten = msk_atten.reshape(-1, self.num_head, self.win_patch ** 2, self.win_patch ** 2)

        msk_atten = self.drop1(msk_atten)
        x = msk_atten @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop2(x)
        # re-shift
        x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch,
                      self.input_dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.nwin[0] * self.win_patch,
                                                                        self.nwin[1] * self.win_patch, self.input_dim)
        x = torch.roll(x, (self.stride, self.stride), dims=(1, 2)).reshape(-1,
                                                                           self.nwin[0], self.win_patch, self.nwin[1],
                                                                           self.win_patch,
                                                                           self.input_dim).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.win_patch ** 2, self.input_dim)
        return x


class ChannelBlock(nn.Module):  # 将win内patch合并，进行窗口内通道聚合
    def __init__(self, blocks=3, variables=3, dim=64, num_head=3, at_drop=0.0, pj_drop=0.0):
        super(ChannelBlock, self).__init__()

        self.blocks = blocks
        self.dim = dim
        self.attention_blocks = nn.ModuleList([])
        self.num_head = num_head
        self.act = nn.GELU()
        for i in range(self.blocks):
            self.attention_blocks.append(
                AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, at_drop=at_drop, pj_drop=pj_drop, variables=variables))
            self.attention_blocks.append(nn.LayerNorm([variables, dim], elementwise_affine=True))
            self.attention_blocks.append(nn.Linear(in_features=dim, out_features=dim))
            self.attention_blocks.append(nn.LayerNorm([variables, dim], elementwise_affine=True))

    def forward(self, x):
        x1 = x
        for i in range(self.blocks):
            x = self.attention_blocks[i * 4](x)
            x = x + x1
            x = self.attention_blocks[i * 4 + 1](x)
            x1 = x
            x = self.attention_blocks[i * 4 + 2](x)
            x = x + x1
            x = self.attention_blocks[i * 4 + 3](x)
            x = self.act(x)
            x1 = x
        return x1


def create_mask(win_patch, stride):
    mask = []
    down = torch.ones([win_patch ** 2, win_patch ** 2])
    right = torch.ones([win_patch ** 2, win_patch ** 2])
    corner = torch.ones([win_patch ** 2, win_patch ** 2])
    for i1 in range(win_patch):
        for i2 in range(win_patch):
            for i3 in range(win_patch):
                for i4 in range(win_patch):
                    if (i1 < win_patch - stride <= i3) or (
                            i1 >= win_patch - stride > i3):
                        down[i1 * win_patch + i2, i3 * win_patch + i4] = 0.0
                    if (i2 < win_patch - stride <= i4) or (
                            i2 >= win_patch - stride > i4):
                        right[i1 * win_patch + i2, i3 * win_patch + i4] = 0.0
                    if (i1 < win_patch - stride and i2 < win_patch - stride) and (
                            i3 >= win_patch - stride or i4 >= win_patch - stride):
                        corner[i1 * win_patch + i2, i3 * win_patch + i4] = 0.0
                    elif (i1 >= win_patch - stride or i2 >= win_patch - stride) and (
                            i3 < win_patch - stride and i4 < win_patch - stride):
                        corner[i1 * win_patch + i2, i3 * win_patch + i4] = 0.0
    mask.append(down)
    mask.append(right)
    mask.append(corner)
    return mask


class SpatioBlock(nn.Module):
    def __init__(self, mask, nwin, win_patch, dim, blocks=3, num_head=3,
                 stride=1, at_drop=0.0, pj_drop=0.0):  # blocks:num of attention per WinBlock
        super(SpatioBlock, self).__init__()
        self.win_patch = win_patch
        self.dim = dim

        self.blocks = blocks
        self.act = nn.GELU()
        self.attention_blocks = nn.ModuleList([])
        for i in range(self.blocks):
            self.attention_blocks.append(AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head,
                                                        at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(nn.Linear(in_features=dim, out_features=dim))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(
                SWAttention(stride=stride, input_dim=dim, hidden_dim=dim, num_head=num_head, mask=mask, nwin=nwin,
                            win_patch=win_patch, at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(nn.Linear(in_features=dim, out_features=dim))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))

    def forward(self, x):
        x1 = x
        for i in range(self.blocks):
            x = self.attention_blocks[i * 6](x)
            x = x + x1
            x = self.attention_blocks[i * 6 + 1](x)
            x1 = x
            x = self.attention_blocks[i * 6 + 2](x)
            x = self.act(x)
            x = x + x1
            x = self.attention_blocks[i * 6 + 3](x)
            x1 = x
            x = self.attention_blocks[i * 6+4](x)
            x = x + x1
            x = self.attention_blocks[i * 6 + 5](x)
            x1 = x
            x = self.attention_blocks[i * 6 + 6](x)
            x = self.act(x)
            x = x + x1
            x = self.attention_blocks[i * 6 + 7](x)
            x1 = x
        return x1


class SpatioBlock_one_win(nn.Module):
    def __init__(self, win_patch, dim, blocks=3, num_head=3, at_drop=0.0, pj_drop=0.0
                 ):  # blocks:num of attention per WinBlock
        super(SpatioBlock_one_win, self).__init__()
        self.win_patch = win_patch
        self.dim = dim
        self.act = nn.GELU()
        self.blocks = blocks
        self.attention_blocks = nn.ModuleList([])
        for i in range(self.blocks):
            self.attention_blocks.append(
                AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(nn.Linear(in_features=dim, out_features=dim))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))

    def forward(self, x):
        x1 = x
        for i in range(self.blocks):
            x = self.attention_blocks[i * 4](x)
            x = x + x1
            x = self.attention_blocks[i * 4 + 1](x)
            x1 = x
            x = self.attention_blocks[i * 4 + 2](x)
            x = x + x1
            x = self.attention_blocks[i * 4 + 3](x)
            x = self.act(x)
            x1 = x
        return x1

def main():
    mask = create_mask(4,2)
    print(mask)

if __name__=='__main__':
    main()
    