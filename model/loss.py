import torch
import torch.nn as nn


def local_loss(p, y, w):
    loss = nn.functional.mse_loss(p, y, reduction='none')

    loss = loss * w.reshape(w.shape[0], 1, 1, 1)
    loss = torch.mean(loss)
    return loss


def around_loss(p, y, w):
    loss = nn.functional.mse_loss(p, y, reduction='none')

    loss = loss * w.reshape(w.shape[0], 1, 1)
    loss = torch.mean(loss)
    return loss


def global_loss(p, y):
    loss = 0
    loss = loss + nn.functional.mse_loss(p, y)
    return loss
