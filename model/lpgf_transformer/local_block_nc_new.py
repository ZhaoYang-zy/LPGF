import torch.nn as nn
import torch
from model.layers.cross_attention import CrossAttention
from model.layers.direction import Direct_TF
from model.layers.space_temporal_block_new import SpatioBlock_one_win, ChannelBlock
from model.layers.fuse import SPADE, GenBlock


class LocalPredictor_new(nn.Module):
    def __init__(self,
                 c_blocks,
                 c_head,
                 s_blocks,
                 s_head,
                 dim,
                 win_patch,
                 out_channel,
                 at_drop=0,
                 pj_drop=0
                 ):
        super(LocalPredictor_new, self).__init__()
        self.win_patch = win_patch
        self.dim = dim
        self.proj1 = nn.Linear(in_features=dim, out_features=dim*2)
        self.proj2 = nn.Linear(in_features=dim, out_features=dim*2)
        self.norm1 = nn.LayerNorm([win_patch**2, dim*2], elementwise_affine=True)
        self.norm2 = nn.LayerNorm([win_patch**2, dim*2], elementwise_affine=True)
        self.around_block = SingleLocalBlock(channel_blocks=c_blocks,
                                             channel_num_head=c_head,
                                             spatio_blocks=s_blocks,
                                             spatio_num_head=s_head,
                                             win_patch=self.win_patch,
                                             dim1=dim * 2,
                                             dim2=dim * 4,
                                             at_drop=at_drop, pj_drop=pj_drop)

        # self.fuse = SPADE(self.dim, self.dim)
        self.cross = CrossAttention(blocks=s_blocks, N1=win_patch**2, N2=win_patch**2, hidden_dim=dim*2, num_head=s_head)

        self.act = nn.GELU()
        self.conv_out = nn.Conv2d(dim*2, out_channel, kernel_size=1, stride=1)

    def forward(self, h):
        B, N, H, W, C = h.shape
        # x先移动再生消
        h1 = h[..., 0:self.dim].reshape(B*N, self.win_patch*self.win_patch, self.dim)
        h2 = h[..., self.dim:].reshape(B*N, self.win_patch*self.win_patch, self.dim)

        h1 = self.norm1(self.proj1(h1)).reshape(B, N, self.win_patch, self.win_patch, self.dim*2)
        h2 = self.norm2(self.proj2(h2))
        x = self.around_block(h1)
        
        x = self.cross(x, h2)

        x = x.permute(0, 2, 1).reshape(-1, self.dim*2, self.win_patch, self.win_patch)
        x = self.act(x)
        x = self.conv_out(x)

        return x


# class LocalPredictor_new(nn.Module):
#     def __init__(self,
#                  c_blocks,
#                  c_head,

#                  s_blocks,
#                  s_head,
#                  dim,
#                  win_patch,
#                  out_channel,
#                  at_drop=0,
#                  pj_drop=0
#                  ):
#         super(LocalPredictor_new, self).__init__()
#         self.win_patch = win_patch

#         self.dim = dim
#         self.around_block = SingleLocalBlock(channel_blocks=c_blocks,
#                                              channel_num_head=c_head,
#                                              spatio_blocks=s_blocks,
#                                              spatio_num_head=s_head,
#                                              win_patch=self.win_patch,
#                                              dim1=dim,
#                                              dim2=dim * 2,
#                                              at_drop=at_drop, pj_drop=pj_drop)

#         # self.fuse = SPADE(self.dim, self.dim)
#         self.fuse = GenBlock(self.dim, self.dim, self.dim)
#         self.prj = nn.Linear(in_features=dim * 2, out_features=dim * 2)
#         self.norm = nn.LayerNorm([win_patch * win_patch, dim], elementwise_affine=True)

#         self.local_block = SingleLocalBlock(channel_blocks=c_blocks,
#                                             channel_num_head=c_head,
#                                             spatio_blocks=s_blocks,
#                                             spatio_num_head=s_blocks,
#                                             win_patch=self.win_patch,
#                                             dim1=dim,
#                                             dim2=dim * 2,
#                                             at_drop=at_drop, pj_drop=pj_drop)
#         self.act = nn.GELU()
#         self.conv_out = nn.Conv2d(dim, out_channel, kernel_size=1, stride=1)

#     def forward(self, h):
#         B, N, H, W, C = h.shape
#         # x先移动再生消
#         h1 = h[..., 0:self.dim].reshape(B, N, self.win_patch, self.win_patch, self.dim)
#         h2 = h[..., self.dim:].permute(0, 1, 4, 2, 3).reshape(-1, self.dim, self.win_patch, self.win_patch)
#         x = self.around_block(h1)
#         x = x.permute(0, 2, 1).reshape(-1, self.dim, self.win_patch, self.win_patch)
#         x = self.fuse(x, h2)
#         x = x.permute(0, 2, 3, 1).reshape(-1, self.win_patch**2, self.dim)
#         x = self.norm(x)
#         x = x.reshape(B, N, self.win_patch, self.win_patch, self.dim)
#         x = self.local_block(x)

#         x = x.permute(0, 2, 1).reshape(-1, self.dim, self.win_patch, self.win_patch)
#         x = self.act(x)
#         x = self.conv_out(x)

#         return x



class AroundPredictor_new(nn.Module):
    def __init__(self, blocks=3, N1=20, N2=16, input_dim=128, hidden_dim=256, output_dim=160, num_head=8):
        super(AroundPredictor_new, self).__init__()
        self.N1 = N1
        self.proj1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.proj2 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.norm1 = nn.LayerNorm([N2, hidden_dim], elementwise_affine=True)
        self.norm2 = nn.LayerNorm([N2, hidden_dim], elementwise_affine=True)
        self.cross1 = CrossAttention(blocks=blocks, N1=N1, N2=N2, hidden_dim=hidden_dim, num_head=num_head)
        self.cross2 = CrossAttention(blocks=blocks, N1=N1, N2=N2, hidden_dim=hidden_dim, num_head=num_head)
        self.act = nn.GELU()
        self.deproj = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, h):
        B, nindex, H, W, C = h.shape
        h_around = h[:, :, :, :, 0:C // 2]
        h_around = h_around.reshape(B * nindex, H * W, C // 2)
        h_local = h[:, :, :, :, C // 2:]
        h_local = h_local.reshape(B * nindex, H * W, C // 2)
        x1 = self.proj1(h_around)
        x1 = self.norm1(x1)
        x3 = torch.zeros(x1.shape[0], self.N1, x1.shape[2]).cuda()
        x3 = self.cross1(x3, x1)
        x2 = self.proj2(h_local)
        x2 = self.norm2(x2)
        x3 = self.cross2(x3, x2)
        x1 = self.act(x3)
        x1 = self.deproj(x3)
        return x1


class SingleLocalBlock(nn.Module):
    def __init__(self,
                 channel_blocks,
                 channel_num_head,
                 spatio_blocks,
                 spatio_num_head,
                 win_patch,
                 dim1,
                 dim2,
                 at_drop=0.0, pj_drop=0.0):
        super(SingleLocalBlock, self).__init__()

        self.dim1 = dim1

        self.win_patch = win_patch
    
        self.spatio_block = SpatioBlock_one_win(win_patch, dim=dim1, blocks=spatio_blocks, num_head=spatio_num_head,
                                                at_drop=at_drop, pj_drop=pj_drop)
        self.norm2 = nn.LayerNorm([win_patch * win_patch, dim1])
        self.proj = nn.Linear(in_features=dim2,
                              out_features=self.dim1)
        self.norm3 = nn.LayerNorm([self.win_patch * self.win_patch, dim1], elementwise_affine=True)

    def forward(self, x):
        B, N, H, W, C = x.shape
        sx = x.reshape(B * N, H * W, C)
        x = x.reshape(-1, H * W, C)
        x1 = self.spatio_block(x)
        x = x + x1
        x = self.norm2(x)
        x = torch.cat([x, sx], dim=2)
        x = self.proj(x)
        x = self.norm3(x)
        return x


class LocalBlock(nn.Module):  # local feature block
    def __init__(self, configs):
        super(LocalBlock, self).__init__()
        self.configs = configs
        self.direct_bias = configs.direct_bias
        self.in_len = configs.in_len
        self.rebuild_start = configs.rebuild_start
        self.pretrain_pred_len = configs.pretrain_pred_len
        self.rebuild_len = self.in_len - self.rebuild_start - 1  # 重建前n-1帧
        self.local_len = self.in_len - self.rebuild_start + self.pretrain_pred_len
        self.radar = configs.patch_size ** 2
        self.dim1 = configs.local_st_hidden
        self.dim2 = configs.local_st_hidden * 2
        self.out_channel = self.radar * self.configs.in_channel
        self.pre_train = configs.pre_train
        self.win_patch = configs.win_patch
        self.around_patch = configs.around_patch
        self.clip_rate = configs.clip_rate
        self.n_around = (self.win_patch + self.around_patch * 2) ** 2 - self.win_patch ** 2
        self.n_local = self.win_patch ** 2
        self.blocks = nn.ModuleList([])
        for i in range(self.configs.local_blocks):
            if i == self.configs.local_blocks - 1:
                self.blocks.append(
                    SingleLocalBlock(
                        channel_blocks=configs.local_channel_blocks,
                        channel_num_head=configs.local_channel_num_head,
                        spatio_blocks=configs.local_spatio_blocks,
                        spatio_num_head=configs.local_spatio_num_head,
                        win_patch=self.win_patch,
                        dim1=self.dim1,
                        dim2=self.dim2,

                        at_drop=configs.at_drop, pj_drop=configs.pj_drop
                    ))
            else:
                self.blocks.append(
                    SingleLocalBlock(
                        channel_blocks=configs.local_channel_blocks,
                        channel_num_head=configs.local_channel_num_head,
                        spatio_blocks=configs.local_spatio_blocks,
                        spatio_num_head=configs.local_spatio_num_head,
                        win_patch=self.win_patch,
                        dim1=self.dim1,
                        dim2=self.dim2,

                        at_drop=configs.at_drop, pj_drop=configs.pj_drop
                    ))
        if self.pre_train:
            self.localPre = LocalPredictor_new(c_blocks=configs.plocal_c_blocks,
                                               c_head=configs.plocal_c_num_head,

                                               s_blocks=configs.plocal_s_blocks,
                                               s_head=configs.plocal_s_num_head,
                                               dim=self.dim1 // 2,

                                               win_patch=configs.win_patch,
                                               out_channel=self.local_len * self.out_channel,
                                               at_drop=0,
                                               pj_drop=0)
            self.aroundPre = AroundPredictor_new(blocks=configs.paround_blocks, N1=self.n_around, N2=self.n_local,
                                                 input_dim=self.dim1 // 2,
                                                 hidden_dim=self.dim1,
                                                 output_dim=self.rebuild_len * self.out_channel,
                                                 num_head=configs.paround_num_head)
            self.direct = Direct_TF(in_channel=self.dim1 // 2, out_channel=self.rebuild_len * 4,
                                    h=self.win_patch,
                                    w=self.win_patch, l=self.rebuild_len)

    def forward(self, local):
        if self.pre_train:
            B, nindex, H, W, C = local.shape
            x = local

            for i in range(self.configs.local_blocks):
                x = x.reshape(B, nindex, H, W, C)

                x = self.blocks[i](x)

            h = x.reshape(B, nindex, H, W, C)
            plocal = self.localPre(h)
            paround = h[:, :, :, :, 0:C // 2]
            direct = self.direct(paround.reshape(B * nindex, H, W, C // 2).permute(0, 3, 1, 2))
            direct = direct.reshape(B, nindex, self.rebuild_len, 4, 1) + self.direct_bias
            #direct = torch.ones(direct.shape).cuda()

            
            paround = self.aroundPre(h)  # B*nindex*rebuild,naround,C
            plocal = plocal.reshape(B, nindex, self.local_len, self.out_channel, H, W).permute(0, 1, 2, 4, 5, 3)

            paround = paround.reshape(B, nindex, self.n_around, self.rebuild_len, self.out_channel
                                      ).permute(0, 1, 3, 2, 4)
            return plocal, paround, direct

        else:
            B, nindex, H, W, C = local.shape
            x = local

            for i in range(self.configs.local_blocks):
                x = x.reshape(B, nindex, H, W, C)
                x = self.blocks[i](x)
            return x
