import logging
import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)

from tools.configs.update import update_configs
from tools.parser import creat_parser
from tools.weights import load_pretrain_continue, print_pretrain_info, save_pretrain_best, save_pretrain_continue
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from tqdm import tqdm
from data import data_sets
from model import model_sets

from data.dataloader import getDataLoader

from tools.log import get_log
from tools.board import create_board, add_to_board




def main():
    # 参数

  

    parse = creat_parser()
    args = parse.parse_args()

    args.pre_train = True
    vali_before = 100000
    update_configs(args=args, exp_name=args.exp_name, model=args.model)

    # model


    model = model_sets[args.model](args)

    model.cuda()
    # opt
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,total_steps=args.epoch, pct_start=0.05, anneal_strategy='cos')

    root = path = os.path.dirname(os.path.dirname(__file__))
    if args.continue_train:
      
        path = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'pretrain_continue{args.exp_describ}.pt')
        epoch_continue, vali_before = load_pretrain_continue(model, optimizer, scheduler, path)
        epoch_continue = epoch_continue+1
    else:
        epoch_continue = 0

    model = torch.nn.DataParallel(model)

    # data
    train_dataset = data_sets[args.exp_name](configs = args,
                                 pre_train=True,
                                 vali=False,
                                 )
    train_dataloader = getDataLoader(batch_size=args.batch_size, dataset=train_dataset, num_workers=args.num_workers)

    # log
    get_log(args.result_root, args.exp_name, args.model, pretrain=True)

    # tensorboard

    board = create_board(args.result_root, args.exp_name, args.model, flush_secs=args.flush_secs, pretrain=True)

    # print model information
    with torch.no_grad():
        print_pretrain_info(model, args=args)

    # train
 
    for epoch in range(epoch_continue, args.epoch):
        train_dataset.vali = False
        with tqdm(total=len(train_dataset)) as t:

            t.set_description('Epoch %i' % epoch)
            model.train()
            Loss = model.module.pre_train_one_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer,
                                                    progress_bar=t)
            Loss = Loss / len(train_dataset) * args.batch_size
            model.eval()

            # progressing
            t.set_postfix({'mean train loss': '{0:1.5f}'.format(Loss),
                           'lr': optimizer.state_dict()['param_groups'][0]['lr']})
        scheduler.step()

        add_to_board(board, 'pretrain_loss', Loss, epoch)
        # save
        if epoch % args.save_interval == 0:
            path_con = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights',
                                    f'pretrain_continue{args.exp_describ}.pt')
            save_pretrain_continue(epoch, model.module, optimizer, scheduler, path_con, vali_before)
            logging.info(f'epoch:{epoch}')
            logging.info(f'train loss:{Loss}')

        # empty cache
        torch.cuda.empty_cache()

        if epoch >= args.save_thresh:  # if vali<best
            # vali
            train_dataset.vali = True
            with tqdm(total=len(train_dataset)) as t:
                t.set_description('vali :')
                with torch.no_grad():
                    vali = model.module.pre_train_vali(model=model, dataloader=train_dataloader, progress_bar=t)
                vali = vali / len(train_dataset) * args.batch_size
                t.set_postfix({'mean vali loss': '{0:1.5f}'.format(vali)})

            add_to_board(board, 'pretrain_vali_loss', vali, epoch)
            logging.info(f'epoch:{epoch}')
            logging.info(f'vali loss:{vali}')

            if vali < vali_before:
                print('vali_before:', vali_before, '>vLoss:', vali)
                vali_before = vali
                path_best = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights',
                                         f'pretrain_best{args.exp_describ}.pt')
                save_pretrain_best(epoch, model.module, optimizer, scheduler, path_best)
                logging.info(f'save best in epoch:{epoch}')

        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
