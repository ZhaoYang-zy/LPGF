import torch
import torch.nn as nn


def local_loss(p, y, w):
    loss = nn.functional.mse_loss(p, y, reduction='none')
    loss = loss * w.reshape(w.shape[0], 1, 1, 1)
    loss = torch.mean(loss)
    return loss


def around_loss(p, y, w, d):
    loss = nn.functional.mse_loss(p, y, reduction='none')
    loss = loss * d
    loss = loss * w.reshape(w.shape[0], 1, 1)
    loss = torch.mean(torch.sum(loss, dim=[2, 3]))

    return loss


def global_loss(p, y):
    loss = nn.functional.mse_loss(p, y)
    return loss

def time_loss(p, y):
    t = torch.linspace(1, 10, 10).reshape(1, -1, 1, 1, 1).cuda()
    loss = nn.functional.mse_loss(p, y, reduction='none' )
    loss = loss*t
    loss = torch.mean(loss)
    return loss


def around_loss_4dr(p, y, w, d, mask):
  
    loss = 0
    loss1 = nn.functional.mse_loss(p, y, reduction='none')
    for i in range(4):
        a = loss1[:, :, :, mask[i], :] * d[:, :, :, i, :].unsqueeze(dim=-2)
        a = a * w.reshape(w.shape[0], 1, 1)
        a = torch.mean(torch.sum(a, dim=[2]))
        loss = loss+a
    return loss
#
# def around_loss_4dr(p, y, w, d, mask):
#     loss = nn.functional.mse_loss(p, y, reduction='none')
#     loss = loss * w.reshape(w.shape[0], 1, 1)
#     loss = torch.mean(torch.sum(loss, dim=[2]))
#
#     return loss
