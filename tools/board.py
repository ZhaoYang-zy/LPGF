import os
from torch.utils.tensorboard import SummaryWriter
import arrow


def create_board(path, exp_name, model, flush_secs=120, pretrain=True):
    now = arrow.now()
    now = now.format('YYYY-MM-DD-HH-mm-ss')

    root = os.path.dirname(os.path.dirname(__file__))
    if pretrain:
        writer = SummaryWriter(log_dir=root + '/' + path + '/' + exp_name + '/' + 'tensorboard' + '/' + model + '_pretrain' + now,
                               flush_secs=flush_secs)

    else:
        writer = SummaryWriter(log_dir=root + '/' + path + '/' + exp_name + '/' + 'tensorboard' + '/' + model + '_train' + now,
                               flush_secs=flush_secs)
    return writer


def add_to_board(writer, name, data0, data1):
    writer.add_scalar(name, data0, data1)
