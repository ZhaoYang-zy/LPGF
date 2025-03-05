import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from matplotlib.colors import ListedColormap
from model.utils.clip_around import creat_mask

from data import data_sets
from tools.weights import load_pretrain

from matplotlib import pyplot as plt

from model import model_sets
from tools.parser import creat_parser
import torch
import  torch.nn as nn

import torchvision.io.image as imreader
# color_list = ['#01a1f7', '#01a1f7', '#0beced', '#01d901', '#019101', '#ffff01', '#e8c101',
#               '#ff9101', '#ff0101', '#d70101', '#c10101', '#ff01f1', '#9701b5', '#c191f1',]
color_list = ['#FFFFFF', '#01a1f7', '#0beced', '#01d901', '#019101', '#ffff01', '#e8c101',
              '#ff9101', '#ff0101', '#d70101', '#c10101', '#ff01f1', '#9701b5', '#c191f1',]
cmap = ListedColormap(color_list)
cmap.set_under('white')


def main():
    # 参数
    test_num = 100
    #test_num = 170
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = True
    update_configs(args=args, exp_name=args.exp_name, model='lpgf_tf_radar_to_wind')
    out_path1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf_radar_to_wind', f'visual/rebuild_true_{test_num}.png')
    
    out_path2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf_radar_to_wind', f'visual/rebuild_pred_{test_num}.png')
    # model

    model = model_sets['lpgf_tf_radar_to_wind'](args)

    # load
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf_radar_to_wind', 'weights', f'train_best{args.exp_describ}.pt')
    load_pretrain(model, path)
    model.cuda()

    

    model.eval()
    train_dataset = data_sets[args.exp_name](
                                 configs = args,
                                 test=False,
                                 pre_train=True,
                                 vali=False,
                                 )
    scale = [train_dataset.mi[2], train_dataset.ma[2]]

    data = train_dataset[test_num]
    X = data[0]
    Y = data[1]
    X = torch.unsqueeze(X, dim=0).cuda()
    Y = torch.unsqueeze(Y, dim=0).cuda()

    paround, around, direct, local = model.forward([X, Y])

    b, n, t, wp, wp, c = local.shape
    
    m = 0
    p = 10
    for i in range(b):
          for j in range(n):
                for k1 in range(wp):
                      for k2 in range(wp):
                            for k3 in range(c):
                                  if local[i,j,args.in_len-args.rebuild_start-2,k1,k2,k3]>m:
                                        m = local[i,j,args.in_len-args.rebuild_start-2,k1,k2,k3]
                                        p = j
    
    local = local[0,p, args.in_len-args.rebuild_start-2, ...].cpu().detach()
    paround = paround[0,p, args.in_len-args.rebuild_start-2, ...].cpu().detach()
    around = around[0,p, args.in_len-args.rebuild_start-2, ...].cpu().detach()
   
    true = torch.zeros([c, args.win_patch+2*args.around_patch, args.win_patch+2*args.around_patch])
    pred = torch.zeros([c, args.win_patch+2*args.around_patch, args.win_patch+2*args.around_patch])
    mask_local, mask_around=creat_mask(args.win_patch, args.around_patch)

    true[:, mask_local] = local.permute(2,0,1).reshape(c, -1)
    true[:, mask_around] = around.permute(1,0)
    pred[:, mask_local] = local.permute(2,0,1).reshape(c, -1)
    pred[:, mask_around] = paround.permute(1,0)

    true = true.reshape(args.in_channel, args.patch_size, args.patch_size, args.win_patch+2*args.around_patch, args.win_patch+2*args.around_patch).permute(3,1,4,2,0).reshape((args.win_patch+2*args.around_patch)*args.patch_size, (args.win_patch+2*args.around_patch)*args.patch_size,args.in_channel)
    pred = pred.reshape(args.in_channel, args.patch_size, args.patch_size, args.win_patch+2*args.around_patch, args.win_patch+2*args.around_patch).permute(3,1,4,2,0).reshape((args.win_patch+2*args.around_patch)*args.patch_size, (args.win_patch+2*args.around_patch)*args.patch_size,args.in_channel)

    true = true[:,:,0]
    pred = pred[:,:,0]
    true = true.numpy()
    pred = pred.numpy()
    pred = pred*(scale[1]-scale[0])+scale[0]
    true = true*(scale[1]-scale[0])+scale[0]
   


    fig = plt.figure()
    if args.exp_name=='movingmnist':
            plt.imshow(true, vmin=scale[0], vmax=scale[1], cmap='gray')
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            fig.savefig(out_path1, bbox_inches='tight')
            plt.imshow(pred, vmin=scale[0], vmax=scale[1], cmap='gray')
            plt.xticks([])
            plt.yticks([])
            fig.savefig(out_path2, bbox_inches='tight')
    else:
            plt.imshow(true, vmin=scale[0], vmax=scale[1], cmap='viridis')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            fig.savefig(out_path1, bbox_inches='tight')
            plt.imshow(pred, vmin=scale[0], vmax=scale[1], cmap='viridis')
            plt.xticks([])
            plt.yticks([])
            fig.savefig(out_path2, bbox_inches='tight')
      
   


if __name__ == '__main__':
    main()
