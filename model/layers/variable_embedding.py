import torch
import torch.nn as nn


class VariableEmbedding(nn.Module):
    def __init__(self, variables, patch2, dim):
        super(VariableEmbedding, self).__init__()

        self.variables = variables

        self.patch2 = patch2
        self.dim = dim
        self.embedding = nn.ModuleList([])
        for i in range(self.variables):
            self.embedding.append(nn.Linear(in_features=self.patch2, out_features=self.dim))

        self.act = nn.LeakyReLU()

    def forward(self, x):
        embeds = []
        for i in range(self.variables):
            embeds.append(self.embedding[i](x[:, i*self.patch2:(i+1)*self.patch2]))
        
        embeds = torch.stack(embeds, dim=1)
        embeds = self.act(embeds)

        # B*patchs*T,variables,rnn_num_hidden
        return embeds




