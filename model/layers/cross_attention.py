import math
import torch
import torch.nn as nn

from model.layers.space_temporal_block_new import FeedForward



class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_head=8, N1=20, N2=16):
        super(CrossAttentionBlock, self).__init__()
        self.num_head = num_head
        self.B = nn.Parameter(torch.randn((N1, N2)))
        self.sdk = torch.tensor(math.sqrt(hidden_dim))
        self.wq = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wk = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wv = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.proj = nn.Linear(in_features=hidden_dim * num_head, out_features=hidden_dim)
        self.sfm = torch.nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        q = self.wq(x1)
        k = self.wk(x2)
        v = self.wv(x2)
        B1, N1, C1 = q.shape
        B2, N2, C2 = k.shape
        q = q.reshape(B1, N1, self.num_head, C1 // self.num_head).permute(0, 2, 1, 3)
        k = k.reshape(B2, N2, self.num_head, C1 // self.num_head).permute(0, 2, 1, 3)
        v = v.reshape(B2, N2, self.num_head, C1 // self.num_head).permute(0, 2, 1, 3)

        atten = q @ torch.transpose(k, -2, -1)
        atten = self.sfm(atten / self.sdk + self.B)

        x = atten @ v
        x = x.transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, blocks=3, N1=20, N2=16, hidden_dim=64, num_head=8):
        super(CrossAttention, self).__init__()
        self.cross_blocks = nn.ModuleList([])
        self.blocks = blocks
        for i in range(self.blocks):
            self.cross_blocks.append(
                CrossAttentionBlock(input_dim=hidden_dim, hidden_dim=hidden_dim, num_head=num_head, N1=N1, N2=N2))
            self.cross_blocks.append(nn.LayerNorm([N1, hidden_dim], elementwise_affine=True))
            self.cross_blocks.append(FeedForward(dim=hidden_dim, hidden_dim=4*hidden_dim))
            self.cross_blocks.append(nn.LayerNorm([N1, hidden_dim], elementwise_affine=True))

    def forward(self, x1, x2):
        x = x1
        for i in range(self.blocks):
            x1 = self.cross_blocks[i * 4](x1, x2)
            x1 = x + x1
            x1 = self.cross_blocks[i*4+1](x1)
            x = x1
            x1 = self.cross_blocks[i*4+2](x1)
            x1 = x+x1
            x1 = self.cross_blocks[i*4+3](x1)
            x = x1
        return x


if __name__ == '__main__':
    x1 = torch.randn((10, 20, 64)).to('cuda:0')
    x2 = torch.randn((10, 16, 64)).to('cuda:0')
    atten = CrossAttention().to('cuda:0')
    out = atten(x1, x2)
