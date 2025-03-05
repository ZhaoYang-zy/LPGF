
import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs
from tools.weights import load_train, print_train_info
os.environ['CUDA_VISIBLE_DEVICES'] = '0'




from tools.parser import creat_parser
import torch
from tqdm import tqdm
from data import data_sets
from model import model_sets
from data.dataloader import getDataLoader

from tools.metric import Metric
from tools.log import get_log, printlog



def main():
    # 参数
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = False


    update_configs(args=args, exp_name=args.exp_name, model=args.model)
    # model
    model = model_sets[args.model](args)
    model.cuda()
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, args.result_root, args.exp_name, args.model, 'weights', f'train_continue{args.exp_describ}.pt')
    load_train(model, path)

    model = torch.nn.DataParallel(model)

    metric = Metric(args.test_metrics)

    # data
    train_dataset = data_sets[args.exp_name](configs=args,
                                 test=True,
                                 pre_train=False,
                                 vali=False,
                                 )
    train_dataloader = getDataLoader(batch_size=args.batch_size, dataset=train_dataset, num_workers=args.num_workers)

    # log
    get_log(args.result_root, args.exp_name, args.model, pretrain=False, test=True)

    # print model information
    with torch.no_grad():
        print_train_info(model, args=args)


    with tqdm(total=len(train_dataset)) as t:

        t.set_description('test :')
        metrics = {}
        n_metrics = {}  # count the number of drop days
        for key in args.test_metrics:
            metrics[key] = [0 for i in range(args.pred_len)]
            n_metrics[key] = [0 for i in range(args.pred_len)]

        model.eval()
        with torch.no_grad():
            model.module.train_test(model=model, dataloader=train_dataloader, progress_bar=t, metric=metric, metrics=metrics,
                                    n_metrics=n_metrics)

        for key in args.test_metrics:
            for i in range(args.pred_len):
                if n_metrics[key][i] > 0:
                    metrics[key][i] = metrics[key][i] / n_metrics[key][i]
            metrics[f'{key}_mean'] = sum(metrics[key])/len(metrics[key])

    printlog(metrics, -1)


if __name__ == '__main__':
    main()
