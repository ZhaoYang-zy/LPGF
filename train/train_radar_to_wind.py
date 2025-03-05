import logging
import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.weights import load_pretrain, load_train_continue, print_train_info, save_train_best, save_train_continue

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from tools.parser import creat_parser
import torch
from tqdm import tqdm

from data.dataloader import getDataLoader
from data import data_sets
from model import model_sets
from tools.metric import Metric
from tools.log import get_log, printlog
from tools.board import create_board, add_to_board

from tools.configs.update import update_configs

def main():
    # 参数

  

    epoch_continue = 0
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = True
    
    update_configs(args=args, exp_name=args.exp_name, model=args.model)

    vali_before = 100000


    # model

    model = model_sets[args.model](args)
   
    model.cuda()

    for name, param in model.named_parameters():
        if not 'aroundPre_new' in name:
            param.requires_grad = False

    # opt
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,total_steps=args.epoch, pct_start=0.05, anneal_strategy='cos')
    

    # load

    root = os.path.dirname(os.path.dirname(__file__))
    if args.continue_train:
        path = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'train_continue{args.exp_describ}.pt')
        epoch_continue, vali_before = load_train_continue(model, optimizer, scheduler, path)
        epoch_continue = epoch_continue+1
    if 'lpgf' in args.model:
        if args.continue_train:
            path = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'train_continue{args.exp_describ}.pt')
            epoch_continue, vali_before = load_train_continue(model, optimizer, scheduler, path)
            epoch_continue = epoch_continue+1
            print('load continue')
        else:
            if not args.train_direct:
                path = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'pretrain_best{args.exp_describ}.pt')
                load_pretrain(model, path)
                epoch_continue = 0
                print('load pretrain')

    model = torch.nn.DataParallel(model)



    # data
   
    train_dataset = data_sets[args.exp_name](configs = args,
                                 pre_train=False,
                                 vali=False,
                                 )
    train_dataloader = getDataLoader(batch_size=args.batch_size, dataset=train_dataset, num_workers=args.num_workers)

    # log
    get_log(args.result_root, args.exp_name, args.model, pretrain=False)

    # board
    board = create_board(args.result_root, args.exp_name, args.model, flush_secs=args.flush_secs, pretrain=False)


    
    # train
    for epoch in range(epoch_continue, args.epoch):
        train_dataset.vali = False
        with tqdm(total=len(train_dataset)) as t:

            t.set_description('Epoch %i' % epoch)
            model.train()
            tLoss = model.module.train_one_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer,
                                                 progress_bar=t)
            model.eval()
            tLoss = tLoss / len(train_dataset) * args.batch_size
            # progressing
            t.set_postfix({'mean train loss': '{0:1.5f}'.format(tLoss),
                           'lr': optimizer.state_dict()['param_groups'][0]['lr']})

        scheduler.step()
        add_to_board(board, 'train_loss', tLoss, epoch)

        # save
        if epoch % args.save_interval == 0:
            path_con = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'train_continue{args.exp_describ}.pt')
            save_train_continue(epoch, model.module, optimizer, scheduler, path_con, vali_before)
            logging.info(f'epoch:{epoch}')
            logging.info(f'train loss:{tLoss}')

        # empty cache
        torch.cuda.empty_cache()

        if epoch >= args.save_thresh:
            # vali

            train_dataset.vali = True
            with tqdm(total=len(train_dataset)) as t:

                t.set_description('vali :')
                metrics = {}
                n_metrics = {}  # count the number of drop days
                for key in args.vali_metrics:
                    metrics[key] = [0 for i in range(args.pred_len)]
                    n_metrics[key] = [0 for i in range(args.pred_len)]
                with torch.no_grad():
                    vLoss = model.module.train_vali(model=model, dataloader=train_dataloader, progress_bar=t,
                                                    )
                vLoss = vLoss / len(train_dataset) * args.batch_size

                t.set_postfix({'mean vali loss': '{0:1.5f}'.format(vLoss)})

                add_to_board(board, 'train_vali_loss', vLoss, epoch)

            logging.info(f'epoch:{epoch}')
            logging.info(f'vali loss:{vLoss}')


            # if vali>best
            if vali_before > vLoss:
                print('vali_before:', vali_before, '>vLoss:', vLoss)
                vali_before = vLoss
                path_best = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'train_best{args.exp_describ}.pt')
                save_train_best(epoch, model.module, optimizer, scheduler, path_best)
                logging.info(f'save best in epoch:{epoch}')

        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
