import os
import sys

import numpy as np
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from scipy.fftpack import fft2, fftshift
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




def main():
    # 参数
    parse = creat_parser()
    args = parse.parse_args()
    args.pre_train = False
    update_configs(args=args, exp_name=args.exp_name, model=args.model)

    
    train_dataset = data_sets[args.exp_name](
                                 configs = args,
                                 test=True,
                                 pre_train=False,
                                 vali=False,
                                 )
    
    scale = [train_dataset.mi[0], train_dataset.ma[0]]

    sq = 0
    pic = 0
    ma = 0
    fre = []
    for test_num in range(0, len(train_dataset), 10):
    
        data = train_dataset[test_num]
        X = data[0]*(scale[1]-scale[0])+scale[0]
        Y = data[1]*(scale[1]-scale[0])+scale[0]
        X = X.numpy()
        Y = Y.numpy()
        for i in range(X.shape[0]):
            if np.max(X[i, 0, ...]>30):
                F_local = fft2(X[i, 0, ...])
                F_local_shifted = fftshift(F_local)
                F_local_shifted = np.log(1 + np.abs(F_local_shifted))
                mid_x, mid_y = F_local_shifted.shape[1]//2, F_local_shifted.shape[0]//2
                y = np.linspace(0, F_local_shifted.shape[0]-1, F_local_shifted.shape[0])
                x = np.linspace(0, F_local_shifted.shape[1]-1, F_local_shifted.shape[1])
                xx, yy=np.meshgrid(x,y)
                L = np.sqrt((xx-mid_x)**2+(yy-mid_y)**2)
                mf_around =np.sum(L*F_local_shifted)/np.sum(L)
                fre.append(mf_around)
                if mf_around>ma:
                    ma = mf_around
                    sq = test_num
                    pic = i
                    print(sq, pic, mf_around)
    print(np.mean(np.stack(fre)))

    data = train_dataset[sq]
    X = data[0]*(scale[1]-scale[0])+scale[0]
    im = plt.imshow(X[pic, 0, ...], cmap=cmap, vmin=scale[0], vmax=scale[1])
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    out_path = os.path.join(os.path.dirname(__file__), 'max.svg')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
                


       
    



if __name__ == '__main__':
    main()
