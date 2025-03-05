from model.earthformer.cuboid_transformer.earthformer import EarthFormer
from model.lpgf_transformer.lpgf_tf import LPGF_TF

from model.mim.mim import MIM
from model.phydnet.phydnet import PhyDNet
from model.predrnn_v2.predrnn_v2 import PredRNNv2
from model.rainformer.Rainformer import RainFormer
from model.simvp.simvp import SimVP
from model.large_model.large import Large
from model.convlstm.convlstm import ConvLSTM_Model
from model.swinlstm.swinlstm import SwinLSTMModel
from model.tau.tau import TAU

model_sets = {
    'convlstm':ConvLSTM_Model,
    'swinlstm':SwinLSTMModel,
    'earthformer': EarthFormer,
    'lpgf_tf': LPGF_TF,
   
    'mim': MIM,
    'phydnet':PhyDNet,
    'predrnnv2':PredRNNv2,
    'rainformer':RainFormer,
    'simvp':SimVP,
    'large':Large,
    'tau':TAU,


}

all = ['model_sets','swinlstm','convlstm', 'crevnet','earthformer','logf_tf', 'lpgf_tf_os', 'mim','phydnet','predrnnv2','rainformer','simvp', 'large', 'tau','lpgf_tf_radar_to_wind']
