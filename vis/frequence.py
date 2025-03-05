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
    args.pre_train = False
    args.return_h = True
    update_configs(args=args, exp_name=args.exp_name, model='lpgf_tf')
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf', 'frequence')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # model

    model = model_sets['lpgf_tf'](args, return_h=True)
 

    # load
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.result_root, args.exp_name, 'lpgf_tf', 'weights', f'train_best{args.exp_describ}.pt')
    load_train(model, path)

    model.cuda()



    model.eval()
    train_dataset = data_sets[args.exp_name](
                                 configs = args,
                                 test=False,
                                 pre_train=False,
                                 vali=False,
                                 )
    test_num = [k for k in range(len(train_dataset)) if k%10==0]


    record_around = []
    record_local = []

    channel = [k for k in range(64) if k%(64//10)==0]
    num = 80
    
    for i in test_num:
        
        local = []
        around = []
        data = train_dataset[i]
        X = data[0]
        
        X = torch.unsqueeze(X, dim=0).cuda()

        h = model.forward([X]).cpu().detach()
        h = h.numpy()
        B, C, H, W = h.shape
        h_around = h[0, :C//2, ...]
        h_local = h[0, C//2:, ...]
        
        for j in range(C//2):
            if i == num and j in channel:
                # 原图像
                plt.imshow(h_around[j,:,:])
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(out_path, f'around_channel_{j}.png'), bbox_inches='tight')
                plt.close()
                plt.imshow(h_local[j,:,:])
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(out_path, f'local_channel_{j}.png'), bbox_inches='tight')
                plt.close()
            # 傅立叶变换
            
            F_around = fft2(h_around[j,:,:])
            F_around_shifted = fftshift(F_around)
            F_around_shifted = np.log(1 + np.abs(F_around_shifted))
            if i == num and j in channel:
                plt.imshow(F_around_shifted, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(out_path, f'fft_around_channel_{j}.png'), bbox_inches='tight')
                plt.close()
            F_local= fft2(h_local[j,:,:])
            F_local_shifted = fftshift(F_local)
            F_local_shifted = np.log(1 + np.abs(F_local_shifted))
            if i == num and j in channel:
                plt.imshow(F_local_shifted, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(out_path, f'fft_local_channel_{j}.png'), bbox_inches='tight')
                plt.close()
            mid_x, mid_y = F_local_shifted.shape[1]//2, F_local_shifted.shape[0]//2
            y = np.linspace(0, F_local_shifted.shape[0]-1, F_local_shifted.shape[0])
            x = np.linspace(0, F_local_shifted.shape[1]-1, F_local_shifted.shape[1])
            X,Y=np.meshgrid(x,y)
            
            L = np.sqrt((X-mid_x)**2+(Y-mid_y)**2)
            mf_around =np.sum(L*F_around_shifted)/np.sum(L)
            mf_local = np.sum(L*F_local_shifted)/np.sum(L)
            around.append(mf_around)
            local.append(mf_local)
           
        record_around.append(np.stack(around, axis=0))
        record_local.append(np.stack(local, axis=0))
    
    record_around = np.stack(record_around, axis=0)
    record_local = np.stack(record_local, axis=0)

    print('around:', np.mean(record_around))
    print('local:', np.mean(record_local))


              


    

if __name__ == '__main__':
    main()
