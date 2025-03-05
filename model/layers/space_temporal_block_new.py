import math

import numpy as np
import torch
import torch.nn as nn
from model.layers.patch_wise_variable_aggregation import AttentionBlock
from torch import nn, einsum
from einops import rearrange, repeat

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

def get_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class SWAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(get_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(get_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x):
        if self.shifted:
            # 左上角移动
            x = self.cyclic_shift(x)


        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


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


# class SWAttention(nn.Module):
#     def __init__(self, stride, input_dim, hidden_dim, num_head, mask, nwin, win_patch, at_drop=0.0, pj_drop=0.0, variables = 16):
#         super(SWAttention, self).__init__()
#         self.B = nn.Parameter(torch.randn((variables, variables)))
#         self.stride = stride
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_head = num_head
#         self.mask = mask
#         self.nwin = nwin
#         self.win_patch = win_patch
#         self.sdk = torch.tensor(math.sqrt(hidden_dim))
#         self.wq = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
#         self.wk = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
#         self.wv = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
#         self.proj = nn.Linear(in_features=hidden_dim * num_head, out_features=hidden_dim)
#         self.sfm = torch.nn.Softmax(dim=-1)
#         self.drop1 = nn.Dropout(at_drop)
#         self.drop2 = nn.Dropout(pj_drop)

#     def forward(self, x):
#         # shift
#         x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch,
#                       self.input_dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.nwin[0] * self.win_patch,
#                                                                         self.nwin[1] * self.win_patch, self.input_dim)
#         x = torch.roll(x, (-self.stride, -self.stride), dims=(1, 2)).reshape(-1,
#                                                                              self.nwin[0], self.win_patch, self.nwin[1],
#                                                                              self.win_patch,
#                                                                              self.input_dim).permute(0, 1, 3, 2, 4, 5)
#         x = x.reshape(-1, self.win_patch ** 2, self.input_dim)

#         q = self.wq(x)
#         k = self.wk(x)
#         v = self.wv(x)
#         B, N, C = q.shape
#         q = q.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)

#         atten = q @ torch.transpose(k, -2, -1)
#         atten = self.sfm(atten / self.sdk + self.B)
#         atten = atten.reshape(-1, self.nwin[0], self.nwin[1], self.num_head, self.win_patch ** 2, self.win_patch ** 2)

#         msk_atten = torch.zeros(atten.shape).cuda()
#         msk_atten[:, 0:self.nwin[0] - 1, 0:self.nwin[1] - 1, :, :, :] = atten[:, 0:self.nwin[0] - 1, 0:self.nwin[1] - 1,
#                                                                         :, :, :]
#         msk_atten[:, -1, 0:self.nwin[1] - 1, :, :, :] = atten[:, -1, 0:self.nwin[1] - 1, :, :, :] * self.mask[0].cuda()
#         msk_atten[:, 0:self.nwin[0] - 1, -1, :, :, :] = atten[:, 0:self.nwin[0] - 1, -1, :, :, :] * self.mask[1].cuda()
#         msk_atten[:, -1, -1, :, :, :] = atten[:, -1, -1, :, :, :] * self.mask[2].cuda()
#         msk_atten = msk_atten.reshape(-1, self.num_head, self.win_patch ** 2, self.win_patch ** 2)

#         msk_atten = self.drop1(msk_atten)
#         x = msk_atten @ v
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.drop2(x)
#         # re-shift
#         x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch,
#                       self.input_dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.nwin[0] * self.win_patch,
#                                                                         self.nwin[1] * self.win_patch, self.input_dim)
#         x = torch.roll(x, (self.stride, self.stride), dims=(1, 2)).reshape(-1,
#                                                                            self.nwin[0], self.win_patch, self.nwin[1],
#                                                                            self.win_patch,
#                                                                            self.input_dim).permute(0, 1, 3, 2, 4, 5)
#         x = x.reshape(-1, self.win_patch ** 2, self.input_dim)
#         return x




class ChannelBlock(nn.Module):  # 将win内patch合并，进行窗口内通道聚合
    def __init__(self, blocks=3, variables=3, dim=64, num_head=3, at_drop=0.0, pj_drop=0.0):
        super(ChannelBlock, self).__init__()

        self.blocks = blocks
        self.dim = dim
        self.attention_blocks = nn.ModuleList([])
        self.num_head = num_head
        for i in range(self.blocks):
            self.attention_blocks.append(
                AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, at_drop=at_drop, pj_drop=pj_drop, variables=variables))
            self.attention_blocks.append(nn.LayerNorm([variables, dim], elementwise_affine=True))
            self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4*dim))
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


# class SpatioBlock(nn.Module):
#     def __init__(self, mask, nwin, win_patch, dim, blocks=3, num_head=3,
#                  stride=1, at_drop=0.0, pj_drop=0.0):  # blocks:num of attention per WinBlock
#         super(SpatioBlock, self).__init__()
#         self.win_patch = win_patch
#         self.dim = dim

#         self.blocks = blocks
#         self.attention_blocks = nn.ModuleList([])
#         for i in range(self.blocks):
#             self.attention_blocks.append(AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head,
#                                                         at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
#             self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
#             self.attention_blocks.append(
#                 SWAttention(stride=stride, input_dim=dim, hidden_dim=dim, num_head=num_head, mask=mask, nwin=nwin,
#                             win_patch=win_patch, at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
#             self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
#             self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4*dim))
#             self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))

#     def forward(self, x):
#         x1 = x
#         for i in range(self.blocks):
#             x = self.attention_blocks[i * 6](x)
#             x = x + x1
#             x = self.attention_blocks[i * 6 + 1](x)
#             x1 = x
#             x = self.attention_blocks[i * 6 + 2](x)
#             x = x + x1
#             x = self.attention_blocks[i * 6 + 3](x)
#             x1 = x
#             x = self.attention_blocks[i * 6 + 4](x)
#             x = x + x1
#             x = self.attention_blocks[i * 6 + 5](x)
#             x1 = x
#         return x1

# class SpatioBlock_one_win(nn.Module):
#     def __init__(self, win_patch, dim, blocks=3, num_head=3, at_drop=0.0, pj_drop=0.0
#                  ):  # blocks:num of attention per WinBlock
#         super(SpatioBlock_one_win, self).__init__()
#         self.win_patch = win_patch
#         self.dim = dim

#         self.blocks = blocks
#         self.attention_blocks = nn.ModuleList([])
#         for i in range(self.blocks):
#             self.attention_blocks.append(
#                 AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
#             self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
#             self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4*dim))
#             self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))

#     def forward(self, x):
#         x1 = x
#         for i in range(self.blocks):
#             x = self.attention_blocks[i * 4](x)
#             x = x + x1
#             x = self.attention_blocks[i * 4 + 1](x)
#             x1 = x
#             x = self.attention_blocks[i * 4 + 2](x)
#             x = x + x1
#             x = self.attention_blocks[i * 4 + 3](x)
#             x1 = x
#         return x1

class SpatioBlock_one_win(nn.Module):
    def __init__(self, win_patch, dim, blocks=3, num_head=3, at_drop=0.0, pj_drop=0.0
                 ):  # blocks:num of attention per WinBlock
        super(SpatioBlock_one_win, self).__init__()
        self.win_patch = win_patch
        self.dim = dim

        self.blocks = blocks
        self.attention_blocks = nn.ModuleList([])
        for i in range(self.blocks):
            self.attention_blocks.append(
                AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, at_drop=at_drop, pj_drop=pj_drop, variables=win_patch**2))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4*dim))
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
            x1 = x
        return x1

class SpatioBlock(nn.Module):
    def __init__(self, mask, nwin, win_patch, dim, blocks=3, num_head=3,
                 stride=1, at_drop=0.0, pj_drop=0.0):  # blocks:num of attention per WinBlock
        super(SpatioBlock, self).__init__()
        self.win_patch = win_patch
        self.dim = dim
        self.nwin = nwin

        self.blocks = blocks
        self.attention_blocks = nn.ModuleList([])
        for i in range(self.blocks):
            self.attention_blocks.append(SWAttention(dim=dim, heads=num_head, head_dim=dim, shifted=True, window_size=win_patch, relative_pos_embedding=False))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4 * dim))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(
                SWAttention(dim=dim, heads=num_head, head_dim=dim, shifted=True, window_size=win_patch, relative_pos_embedding=False))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))
            self.attention_blocks.append(FeedForward(dim=dim, hidden_dim=4*dim))
            self.attention_blocks.append(nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True))

    def forward(self, x):
        x1 = x
        for i in range(self.blocks):
            x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch, self.dim).permute(0,1,3,2,4,5).reshape(-1, self.nwin[0]*self.win_patch, self.nwin[1]*self.win_patch, self.dim)
            x = self.attention_blocks[i * 8](x)
            x = x.reshape(-1, self.nwin[0], self.win_patch, self.nwin[1], self.win_patch, self.dim).permute(0,1,3,2,4,5).reshape(-1, self.win_patch**2, self.dim)
            x = x + x1
            x = self.attention_blocks[i * 8 + 1](x)
            x1 = x
            x = self.attention_blocks[i * 8 + 2](x)
            x = x+x1
            x = self.attention_blocks[i * 8 + 3](x)
            x = x.reshape(-1, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch, self.dim).permute(0, 1, 3, 2,
                                                                                                            4,
                                                                                                            5).reshape(
                -1, self.nwin[0] * self.win_patch, self.nwin[1] * self.win_patch, self.dim)
            x = self.attention_blocks[i * 8+4](x)
            x = x.reshape(-1, self.nwin[0], self.win_patch, self.nwin[1], self.win_patch, self.dim).permute(0, 1, 3, 2,
                                                                                                            4,
                                                                                                            5).reshape(
                -1, self.win_patch ** 2, self.dim)
            x = x + x1
            x = self.attention_blocks[i * 8 + 5](x)
            x1 = x
            x = self.attention_blocks[i * 8 + 6](x)
            x = x + x1
            x = self.attention_blocks[i * 8 + 7](x)
            x1 = x
        return x1



def main():
    mask = create_mask(4,2)
    print(mask)

if __name__=='__main__':
    main()
