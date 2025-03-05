import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from scipy.fftpack import fft2, fftshift
import numpy as np
from matplotlib.colors import ListedColormap
from model.utils.clip_around import creat_mask

from data import data_sets
from tools.weights import load_train

from matplotlib import pyplot as plt

from model import model_sets
from tools.parser import creat_parser
import torch
import  torch.nn as nn

import torchvision.io.image as imreader
color_list = ['#01a1f7', '#01a1f7', '#0beced', '#01d901', '#019101', '#ffff01', '#e8c101',
              '#ff9101', '#ff0101', '#d70101', '#c10101', '#ff01f1', '#9701b5', '#c191f1',]
cmap = ListedColormap(color_list)
cmap.set_under('white')


def main():
    # 参数
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = True
    update_configs(args=args, exp_name=args.exp_name, model='lpgf_tf')


    # model

    model = model_sets['lpgf_tf'](args, return_h=True)
 

    # load
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf', 'weights', f'pretrain_best{args.exp_describ}.pt')
    load_train(model, path)

    model.cuda()

    names = ['lu', 'ru', 'ld', 'rd']

    model.eval()
    train_dataset = data_sets[args.exp_name](
                                 configs = args,
                                 test=False,
                                 pre_train=True,
                                 vali=False,
                                 )
    test_num = [k for k in range(2000)]

    Dir = [0 for i in range(4)]

    for i in test_num:
        
        X = train_dataset[i]
        
        X = torch.unsqueeze(X, dim=0).cuda()

        plocal, local, paround, around, direct = model.forward(X)
        direct = direct.cpu().detach().numpy()
        dir = np.argmax(direct, axis=3)
        for j in range(4):
            Dir[j] += np.sum(dir==j)
    
    plt.pie(Dir, labels=names, labeldistance=1.15, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, autopct='%1.1f%%')

    path = os.path.dirname(__file__)
    plt.savefig(os.path.join(path, 'direct.svg'), bbox_inches='tight')

    # Dir = [32202, 2887, 33209, 3438]
    # names = ['nw', 'ne', 'sw', 'se']
    # plt.pie(Dir, labels=names, labeldistance=1.15, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, autopct='%1.1f%%')

    # path = os.path.dirname(__file__)
    # plt.savefig(os.path.join(path, 'direct.svg'), bbox_inches='tight')
       
        
        


              


    

if __name__ == '__main__':
    main()
