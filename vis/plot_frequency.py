from matplotlib import pyplot as plt
import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
sys.path.append(path)
from tools.configs.update import update_configs
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams

name = ['direct_bias=0', 'direct_bias=0.1', 'direct_bias=0.2', 'direct_bias=0.3', 'direct_bias=0.5', 'w/o direct']
data = [1.315, 1.223, 1.224, 1.200, 1.125, 1.024]

fig, ax = plt.subplots()
for i in range(len(name)):
    ax.bar(i, data[i], label=name[i])
for i, v in enumerate(data):
    plt.text(i, v, v, ha='center', va='bottom')
plt.xlabel('Settings')
plt.ylabel('R_IWAF')
plt.xticks([])
plt.ylim(0.8, 1.5)
plt.legend()
path = os.path.dirname(__file__)
plt.savefig(os.path.join(path, 'frequency.svg'), bbox_inches='tight')
