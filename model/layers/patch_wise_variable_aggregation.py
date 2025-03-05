import torch.nn as nn
import math

import torch

class AttentionBlock(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_head=8, at_drop=0.0, pj_drop=0.0, variables=3):
        super(AttentionBlock, self).__init__()
        self.num_head = num_head
        self.B = nn.Parameter(torch.randn((variables, variables)))
        self.sdk = torch.tensor(math.sqrt(hidden_dim))
        self.wq = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wk = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.wv = nn.Linear(in_features=input_dim, out_features=hidden_dim * num_head)
        self.proj = nn.Linear(in_features=hidden_dim * num_head, out_features=hidden_dim)
        self.sfm = torch.nn.Softmax(dim=-1)
        self.drop1 = nn.Dropout(at_drop)
        self.drop2 = nn.Dropout(pj_drop)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)

        atten = q @ torch.transpose(k, -2, -1)
        atten = self.sfm(atten / self.sdk + self.B)
        atten = self.drop1(atten)
        x = atten @ v
        x = x.transpose(1, 2).reshape(B, N, C)  # merge head
        x = self.proj(x)
        x = self.drop2(x)
        return x


class VariableAggregation(nn.Module):
    def __init__(self, blocks=3, variables=3, dim=64, num_head=3):
        super(VariableAggregation, self).__init__()

        self.blocks = blocks
        self.dim = dim
        self.attention_blocks = nn.ModuleList([])
        self.num_head = num_head
        for i in range(self.blocks):
            self.attention_blocks.append(
                AttentionBlock(input_dim=dim, hidden_dim=dim, num_head=num_head, variables=variables))
            self.attention_blocks.append(nn.LayerNorm([variables, dim], elementwise_affine=True))

    def forward(self, x):
        x1 = x
        for i in range(self.blocks):
            x = self.attention_blocks[i * 2](x)
            x = x + x1
            x = self.attention_blocks[i * 2 + 1](x)
            x1 = x
        x1 = torch.mean(x1, dim=1)
        return x1

class VariableAggregation_Linear(nn.Module):

    def __init__(self, blocks=3, variables=2, dim=64, num_head=3):
        super(VariableAggregation_Linear, self).__init__()
        self.dim = dim
        self.variables = variables
        self.proj = nn.Linear(in_features=self.variables*dim, out_features=dim)
        self.act = nn.SiLU()

    def forward(self, x):

        x = x.reshape(-1, self.variables*self.dim)
        x = self.proj(x)
        x = self.act(x)
        return x


if __name__ == '__main__':
    x1 = torch.randn((10, 3, 64)).to('cuda:0')
    atten = VariableAggregation().to('cuda:0')
    out = atten(x1)
    print(out.shape)
