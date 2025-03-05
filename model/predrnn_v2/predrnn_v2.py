import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.rnn_sampling import schedule_sampling, reserve_schedule_sampling_exp
from model.loss_direct import global_loss
from model.predrnn_v2.predrnnv2_modules import SpatioTemporalLSTMCellv2


class PredRNNv2(nn.Module):
    r"""PredRNNv2 Model

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, configs):
        super(PredRNNv2, self).__init__()

        self.configs = configs

        self.num_updates = 0

        self.input_size = configs.input_size
        self.rnn_num_layers = configs.rnn_num_layers
        self.rnn_num_hidden = configs.rnn_num_hidden
        self.in_channel = configs.in_channel
        self.frame_channel = configs.in_channel*configs.patch_size * configs.patch_size
        self.out_channel = configs.out_channel*configs.patch_size * configs.patch_size
        self.in_len = configs.in_len
        self.pred_len = configs.pred_len
        cell_list = []
        self.patch_size = configs.patch_size
        self.height = configs.input_size[0] // configs.patch_size
        self.width = configs.input_size[1] // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(configs.rnn_num_layers):
            in_channel = self.frame_channel if i == 0 else configs.rnn_num_hidden
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, configs.rnn_num_hidden, self.height, self.width,
                                         configs.rnn_filter_size, configs.rnn_stride, configs.rnn_layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(configs.rnn_num_hidden, self.out_channel, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.act = nn.Sigmoid()
        # shared adapter
        adapter_num_hidden = configs.rnn_num_hidden
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true=None, return_loss=False):
        # [batch, length, channel, height, width] -> [batch, length, channel, height, width]

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
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.rnn_num_layers):
            zeros = torch.zeros(
                [batch, self.rnn_num_hidden, height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch, self.rnn_num_hidden, height, width]).cuda()

        for t in range(self.configs.in_len + self.configs.pred_len - 1):
            # reverse schedule sampling
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

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.rnn_num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.rnn_num_layers - 1])
            next_frames.append(x_gen)

            # decoupling loss
            if return_loss:
                for i in range(0, self.rnn_num_layers):
                    decouple_loss.append(torch.mean(torch.abs(
                        torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if return_loss:
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # [length, batch, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).reshape(T - 1, B, C, self.patch_size, self.patch_size,
                                                              self.height,
                                                              self.width).permute(1, 0, 2, 5, 3, 6,
                                                                                  4).reshape(B, T - 1, C, H, W)
        if return_loss:
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + \
                   self.configs.decouple_beta * decouple_loss
            return next_frames[:, self.in_len - 1:, ...], loss
        else:
            return next_frames[:, self.in_len - 1:, ...]

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
