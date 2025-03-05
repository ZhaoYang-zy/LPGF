import math
import torch
import torch.nn as nn
from model.layers.space_temporal_block_new import create_mask
from model.layers.fuse import SPADE, GenBlock, ProjBlock
from model.layers.lpgf_simvp import GroupConv2d, gInception_ST,sampling_generator,ConvSC


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                   act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
       
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class FinalDecoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_conv_layers=1,
                 up_scale=4,
                
                 ):
        super(FinalDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_conv_layers = num_conv_layers
        self.up_scale = up_scale


        up_conv = []
        for i in range(int(math.log2(self.up_scale))):
            up_conv.append(nn.ConvTranspose2d(in_channels=in_dim//2**i, out_channels=in_dim//2**(i+1), kernel_size=4, stride=2, padding=1))
            up_conv.append(nn.SiLU())
            up_conv.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=in_dim//2**(i+1), out_channels=in_dim//2**(i+1)))
            up_conv.append(nn.SiLU())
        self.up_conv = nn.Sequential(*up_conv)

        # conv_block = []
        # for i in range(num_conv_layers):
        #     conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=in_dim//self.up_scale, out_channels=in_dim//self.up_scale))
        #     conv_block.append(nn.SiLU())
        # self.conv_block = nn.Sequential(*conv_block)

        self.proj_final = nn.Conv2d(kernel_size=(1, 1), in_channels=in_dim//self.up_scale, out_channels=out_dim)

        # conv_block = []
        # for i in range(num_conv_layers):
        #     conv_block.append(nn.Conv2d(kernel_size=(3, 3), padding=(1, 1), in_channels=in_dim, out_channels=in_dim))
        #     conv_block.append(nn.SiLU())
        # self.conv_block = nn.Sequential(*conv_block)

        # self.proj_final = nn.Conv2d(kernel_size=(1, 1), in_channels=in_dim, out_channels=out_dim)


        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.up_conv(x)
        # if self.num_conv_layers > 0:
        #     B, C, H, W = x.shape
        #     x = self.conv_block(x)
        x = self.proj_final(x)
      
        return x



class GlobalBlock(nn.Module):
    #def __init__(self, configs, incep_ker_high=[5, 5, 5, 5], incep_ker_low=[3, 3, 3, 3]):
    def __init__(self, configs, incep_ker_high=[3, 3, 3, 3], incep_ker_low=[5, 5, 5, 5]):
        super(GlobalBlock, self).__init__()
        self.configs = configs
        self.global_blocks = configs.global_blocks

        self.position = configs.position
        groups = configs.groups

        self.input_size = configs.input_size
        self.hdim = configs.local_st_hidden
        self.dim1 = configs.global_num_hidden
        self.dim2 = self.hdim
        self.out_channel = configs.out_channel
        self.win_patch = self.configs.win_patch
        self.patch_size = self.configs.patch_size
        self.dim4 = configs.pred_len * self.configs.out_channel
        
        self.nwin = []
        self.nwin.append(self.input_size[0] // (self.win_patch * self.patch_size))
        self.nwin.append(self.input_size[1] // (self.win_patch * self.patch_size))
        self.mask = create_mask(win_patch=self.win_patch, stride=configs.shift_stride,
                                )
        N2 = self.configs.spatio_blocks
        self.N2 = N2
        enc_layers = [gInception_ST(
            self.dim1+ self.dim2 // 2, self.dim1, self.dim1, incep_ker=incep_ker_high, groups=groups)]
        for i in range(1, N2 - 1):
            if i <= (self.N2-1)//4:
                enc_layers.append(
                    gInception_ST(self.dim1+ self.dim2 // 2, self.dim1, self.dim1,
                                incep_ker=incep_ker_high, groups=groups))
            else:
                if i == (self.N2-1)//4+1:
                    enc_layers.append(
                        gInception_ST(self.dim1+ self.dim2 // 2, self.dim1, self.dim1,
                                    incep_ker=incep_ker_low, groups=groups))
                else:
                    enc_layers.append(
                        gInception_ST(self.dim1 + self.dim2 // 2, self.dim1, self.dim1,
                                    incep_ker=incep_ker_low, groups=groups))
        enc_layers.append(
            gInception_ST(self.dim1 + self.dim2 // 2, self.dim1, self.dim1,
                          incep_ker=incep_ker_low, groups=groups))
    
        dec_layers = [
            gInception_ST(self.dim1+ self.dim2 // 2, self.dim1*2, self.dim1,
                          incep_ker=incep_ker_low, groups=groups)]
        for i in range(1, N2 - 1):
            j = N2-1-i
            if j<=(self.N2-1)//4:
                dec_layers.append(
                    gInception_ST(self.dim1*2+ self.dim2 // 2, self.dim1*2, self.dim1,
                                incep_ker=incep_ker_high, groups=groups))
            else:
                dec_layers.append(
                    gInception_ST(self.dim1*2+ self.dim2 // 2, self.dim1*2, self.dim1,
                                incep_ker=incep_ker_low, groups=groups))
        dec_layers.append(
            gInception_ST(self.dim1*2+ self.dim2 // 2, self.dim1*2, self.dim1,
                          incep_ker=incep_ker_high, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

        self.conv_final = FinalDecoder(in_dim=self.dim1*2, out_dim=self.dim4, num_conv_layers=2, up_scale=self.patch_size)
        #self.conv_final = Decoder(self.dim1, self.dim4, 4, 3, False)


    def forward(self, x, h):
        B = x.shape[0]
        
        h = h.reshape(B, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, self.nwin[0]*self.win_patch, self.nwin[1]*self.win_patch)
        h_around = h[:, 0:self.hdim//2, ...]
        h_local = h[:, self.hdim//2:, ...]
        x = x.reshape(B, self.nwin[0], self.nwin[1], self.win_patch, self.win_patch, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, self.nwin[0]*self.win_patch, self.nwin[1]*self.win_patch)

        B, C, H, W = x.shape
        # encoder
        skips = []
        z = x
        f = z
        # z = self.enc[0](torch.cat([z, h_local], dim=1))
        skips.append(z)
        for i in range(self.N2):
            if i<=(self.N2-1)//4:
                z = self.enc[i](torch.cat([z, h_around], dim=1))
            else:
                z = self.enc[i](torch.cat([z, h_local], dim=1))
            if i < self.N2 - 1:
                skips.append(z)
        # decoder


        z = self.dec[0](torch.cat([z, h_local], dim=1))
        #z = self.dec[0](torch.cat([z, h_around], dim=1))


        for i in range(1, self.N2):
            j = self.N2-1-i
            if j <= (self.N2-1)//4:
                z = self.dec[i](torch.cat([z, skips[-i], h_around], dim=1))
            else:
                z = self.dec[i](torch.cat([z, skips[-i], h_local], dim=1))
        x = z

        x = torch.cat([x, f], dim=1)
        
        x = self.conv_final(x)
        
        x = x.reshape(B, self.configs.pred_len, self.out_channel, self.input_size[0],
                                                 self.input_size[1])

        # x = x.reshape(B, self.configs.pred_len, self.out_channel, self.patch_size, self.patch_size,
        #               self.nwin[0] * self.win_patch, self.nwin[1] * self.win_patch). \
        #     permute(0, 1, 2, 5, 3, 6, 4).reshape(B, self.configs.pred_len, self.out_channel, self.input_size[0],
        #                                          self.input_size[1])

        return x


