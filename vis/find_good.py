import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from data import data_sets
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tools.parser import creat_parser
import torch
from model import model_sets
import torchvision.io.image as imreader

from tools.weights import load_train

# color_list = ['#01a1f7', '#01a1f7', '#0beced', '#01d901', '#019101', '#ffff01', '#e8c101',
#               '#ff9101', '#ff0101', '#d70101', '#c10101', '#ff01f1', '#9701b5', '#c191f1',]
color_list = ['#FFFFFF', '#01a1f7', '#0beced', '#01d901', '#019101', '#ffff01', '#e8c101',
              '#ff9101', '#ff0101', '#d70101', '#c10101', '#ff01f1', '#9701b5', '#c191f1',]
cmap = ListedColormap(color_list)
cmap.set_under('white')


def csi_30(pre, true):
        pre = pre>30
        true = true>30
        TP = (true & pre).sum()
        FN = (true & ~pre).sum()
        FP = (~true & pre).sum()
        TN = (~true & ~pre).sum()
        d = TP+FN+FP
        u = TP
        if TP+FN > 0:
            csi = u/d
            return 1, csi.item()
        else:
            return 0, 0




def main():
    # 参数
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = False
    update_configs(args=args, exp_name=args.exp_name, model=args.model)

    
    # model

    model = model_sets[args.model](args)

    # load
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, args.model, 'weights', f'train_continue{args.exp_describ}.pt')
    load_train(model, path)

    model.cuda()


    model.eval()
    train_dataset = data_sets[args.exp_name](
                                 configs = args,
                                 test=True,
                                 pre_train=False,
                                 vali=False,
                                 )
    
    

    best = 0
    MSE = 1000000
    CSI = 0

    for test_num in range(len(train_dataset)):
    
        data = train_dataset[test_num]
        X = data[0]
        Y = data[1]

        X = torch.unsqueeze(X, dim=0).cuda()
        Y = torch.unsqueeze(Y, dim=0).cuda()

        # pred = model([X,Y]).cpu().detach()

       
        Y = Y.cpu().detach()
        Y = Y*80

        strength = torch.sum(Y>30)/torch.sum(Y>-1000)
   
        if 0.05 < strength:
            # _, csi = csi_30(Y[-1, ...], pred[-1, ...])
            # mse = torch.nn.functional.mse_loss(Y, pred)
            # if MSE>mse:
            #     MSE = mse
            #     best = test_num
            #     print(best)
            
                
            best = test_num
            print(best)


       
    



if __name__ == '__main__':
    main()
