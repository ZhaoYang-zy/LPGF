import os
import sys

import numpy as np
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

ax = plt.axes([0, 0, 1, 1], frameon=False)
 
# 生成数据
data = np.random.rand(10, 10)
 
# 在不可见轴上绘制图形
# im = plt.imshow(data, cmap='viridis', vmin=0, vmax=70)
im = plt.imshow(data, cmap=cmap, vmin=0, vmax=70)
 
# 创建colorbar
plt.colorbar(im, ax=ax)
 
# 关闭当前轴的显示
ax.set_visible(False)
 

path = os.path.dirname(__file__)
plt.savefig(os.path.join(path, 'colorbar.svg'), bbox_inches='tight')