import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        print(type(true))
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

    interval = 1

    test_num = 1022
    #test_num = 126
    


    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = False
    update_configs(args=args, exp_name=args.exp_name, model=args.model)

    
    out_path1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, args.model, f'visual/{test_num}')
    if not os.path.exists(out_path1):
        os.makedirs(out_path1)
    
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
    
    scale = [train_dataset.mi[0], train_dataset.ma[0]]
    
    data = train_dataset[test_num]

    X = data[0]
    Y = data[1]

    X = torch.unsqueeze(X, dim=0).cuda()
    Y = torch.unsqueeze(Y, dim=0).cuda()

    pred = model([X,Y]).cpu().detach()

    pred = torch.squeeze(pred, dim=0)

    X = torch.squeeze(X, dim=0).cpu().detach()
    Y = torch.squeeze(Y, dim=0).cpu().detach()


    true = torch.cat([X[:,0,...], Y[:,0,...]], dim=0)
    result = torch.cat([X[:,0,...], pred[:,0,...]], dim=0)
    result = result.numpy()
    true = true.numpy()

    result = result[0::interval, ...]
    true = true[0::interval, ...]

    result = result*(scale[1]-scale[0])+scale[0]
    true = true*(scale[1]-scale[0])+scale[0]

    _, csi = csi_30(result[-1, ...], true[-1, ...])
    print(csi)

    
    im = None
    for t in range(true.shape[0]):
        out_path = os.path.join(out_path1, f'true{(t) * interval+1}.png')

        if args.exp_name=='movingmnist':
            im = plt.imshow(true[t, ...], cmap='gray', vmin=scale[0], vmax=scale[1])
        else:
            im = plt.imshow(true[t, ...], cmap=cmap, vmin=scale[0], vmax=scale[1])
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


    
    for t in range(result.shape[0]):
        out_path = os.path.join(out_path1, f'pred{(t) * interval+1}.png')

        if args.exp_name=='movingmnist':
            im = plt.imshow(result[t, ...], cmap='gray', vmin=scale[0], vmax=scale[1])
        else:
            im = plt.imshow(result[t, ...], cmap=cmap, vmin=scale[0], vmax=scale[1])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
   



if __name__ == '__main__':
    main()
