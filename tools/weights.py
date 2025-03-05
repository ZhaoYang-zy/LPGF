import logging
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
def load_pretrain_continue(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    vali_before = checkpoint['vali_before']
    print(f'load epoch:{epoch}, {path}')
    return epoch,vali_before


def print_pretrain_info(model, args):
    dic = vars(args)
    print('exp configs:')
    logging.info('exp configs:')
    for key, value in dic.items():
        print(key, ':', value)
        logging.info(f'{key}:{value}')

    inputs = torch.randn(
        (1, args.in_len + args.pretrain_pred_len, args.in_channel,
         args.input_size[0],
         args.input_size[1])).cuda(),

    print(model.__repr__())
    flops = FlopCountAnalysis(model, inputs)
    flops = flop_count_table(flops)
    print(flops)


def save_pretrain_continue(epoch, model, optimizer, scheduler, path, vali_before):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'vali_before':vali_before,
    }
    torch.save(checkpoint, path)
    
def save_pretrain_best(epoch, model, optimizer, scheduler, path):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def print_train_info(model, args):
    dic = vars(args)
    print('exp configs:')
    logging.info('exp configs:')
    for key, value in dic.items():
        print(key, ':', value)
        logging.info(f'{key}:{value}')

    inputs = [
        torch.randn(
            (1, args.in_len, args.in_channel,
             args.input_size[0],
             args.input_size[1])).cuda(),
                     torch.randn(
            (1, args.pred_len, args.in_channel,
             args.input_size[0],
             args.input_size[1])).cuda(),
        torch.ones(1, args.pred_len, args.input_size[0] // args.patch_size,
                   args.input_size[1] // args.patch_size).cuda()]
    print(model.__repr__())
    flops = FlopCountAnalysis(model, inputs)
    flops = flop_count_table(flops)
    print(flops)


def save_train_continue(epoch, model, optimizer, scheduler, path, vali_before):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'vali_before':vali_before,
    }
    torch.save(checkpoint, path)


def load_train_continue(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    vali_before = checkpoint['vali_before']
    print(f'load epoch:{epoch}, {path}')
    return epoch, vali_before


def load_pretrain(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'], strict=False)


def save_train_best(epoch, model, optimizer, scheduler, path):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def load_train(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
