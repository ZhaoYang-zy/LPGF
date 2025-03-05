import torch
import torch.nn as nn
from model.loss_direct import global_loss
from model.modules import SpatioTemporalLSTMCell, MIMBlock, MIMN
from model.layers.rnn_sampling import reserve_schedule_sampling_exp, schedule_sampling

class MIM(nn.Module):
    r"""MIM Model

    Implementation of `Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    <https://arxiv.org/abs/1811.07490>`_.

    """

    def __init__(self, configs):
        super(MIM, self).__init__()
        H = configs.input_size[0]
        W = configs.input_size[1]
        C = configs.in_channel

        self.num_updates = 0
        self.configs = configs
        self.patch_size = configs.patch_size
        num_layers = configs.rnn_num_layers
        num_hidden = configs.rnn_num_hidden
        self.frame_channel = C*configs.patch_size * configs.patch_size
        self.out_channel = configs.out_channel*configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        stlstm_layer, stlstm_layer_diff = [], []

        self.height = H // configs.patch_size
        self.width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden
           
            if i < 1:
                stlstm_layer.append(
                    SpatioTemporalLSTMCell(in_channel, num_hidden, self.height, self.width,
                                           configs.rnn_filter_size, configs.rnn_stride, configs.rnn_layer_norm))
            else:
                stlstm_layer.append(
                    MIMBlock(in_channel, num_hidden, self.height, self.width, configs.rnn_filter_size,
                             configs.rnn_stride, configs.rnn_layer_norm))
        
        for i in range(num_layers-1):
            stlstm_layer_diff.append(
                MIMN(num_hidden, num_hidden, self.height, self.width, configs.rnn_filter_size,
                     configs.rnn_stride, configs.rnn_layer_norm))
            
        self.stlstm_layer = nn.ModuleList(stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(stlstm_layer_diff)
        self.conv_last = nn.Conv2d(num_hidden, self.out_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true=None, return_loss=False):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]

        frames_tensor = torch.cat([frames_tensor[0], frames_tensor[1]], dim=1)
        B, T, C, H, W = frames_tensor.shape
        frames = frames_tensor.reshape(B, T, C, self.height, self.patch_size, self.width,
                                       self.patch_size).permute(0, 1, 2, 4, 6, 3, 5).reshape(B, T,
                                                                                             C * self.patch_size ** 2,
                                                                                             self.height, self.width)
        
        if return_loss:
            mask_true = mask_true.cuda()


        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        hidden_state_diff = []
        cell_state_diff = []
       

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden, height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        st_memory = torch.zeros(
            [batch, self.num_hidden, height, width]).cuda()

        for t in range(self.configs.in_len + self.configs.pred_len - 1):
           
            
            if return_loss:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.in_len:
                    net = frames[:, t]
                else:
                    net = x_gen

            preh = h_t[0]

            
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](net, h_t[0], c_t[0], st_memory)

            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            h_t[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(h_t[i - 1]), None, None)

                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](
                    h_t[i - 1], hidden_state_diff[i-1], h_t[i], c_t[i], st_memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        # [length, batch, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).reshape(T - 1, B, C, self.patch_size, self.patch_size,
                                                              self.height,
                                                              self.width).permute(1, 0, 2, 5, 3, 6,
                                                                                  4).reshape(B, T - 1, C, H, W)
        if return_loss:
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            return next_frames[:, self.configs.in_len - 1:, ...], loss
        else:
            return next_frames[:, self.configs.in_len - 1:, ...]

        
    
    def train_one_epoch(self, model, dataloader, optimizer, progress_bar):
        Loss = 0
        for i, true in enumerate(dataloader):
            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()

            optimizer.zero_grad()
            real_input_flag = reserve_schedule_sampling_exp(
                    self.num_updates, self.configs.batch_size, self.configs)
            pred, loss = model.forward([true[0], true[1]], mask_true=real_input_flag, return_loss=True)
            
            self.num_updates = self.num_updates+1

            loss = torch.mean(loss)
            Loss2 = Loss2 + loss.item()
            loss.backward()


            # for name, param in self.named_parameters():
            #     if torch.isnan(param.grad).any():
            #         print(name)
            #         idex = param.grad[~torch.isnan(param.grad)]
            #         if len(idex) > 0:
            #             print(torch.max(abs(idex)))

            optimizer.step()
            Loss = Loss + Loss2
            progress_bar.set_postfix(
                {'loss': '{0:1.5f}'.format(Loss2), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            progress_bar.update(dataloader.batch_size)

        return Loss

    def train_vali(self, model, dataloader, progress_bar, metric, metrics, n_metrics):  # 以loss为选择指标,求所有leadtime的平均值
        Loss = 0
        for i, true in enumerate(dataloader):

            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()

         
            pred = model.forward([true[0], true[1]])
            loss = global_loss(pred, true[1])
            Loss2 = Loss2 + loss.item()

            # Denorm
            pred = dataloader.dataset.denorm(pred.cpu().detach())
            y = dataloader.dataset.denorm(true[1].cpu().detach())

            # B, C, H, W
            for j in range(self.configs.batch_size):
                for k in range(self.configs.pred_len):
                    metric.get_metrics(pred[j, k, ...], y[j, k, ...], metrics, n_metrics, k)

            Loss = Loss + Loss2
            progress_bar.set_postfix({'loss': '{0:1.5f}'.format(Loss2)})
            progress_bar.update(dataloader.batch_size)

        return Loss

    def train_test(self, model, dataloader, progress_bar, metric, metrics, n_metrics):

        for i, true in enumerate(dataloader):

            for k in range(2):
                true[k] = true[k].cuda()

            pred = model.forward([true[0], true[1]], return_loss=False)

            # Denorm
            pred = dataloader.dataset.denorm(pred.cpu().detach())
            y = dataloader.dataset.denorm(true[1].cpu().detach())

            # B, C, H, W
            for j in range(self.configs.batch_size):
                for k in range(self.configs.pred_len):
                    metric.get_metrics(pred[j, k, ...], y[j, k, ...], metrics, n_metrics, k)
            progress_bar.update(dataloader.batch_size)
