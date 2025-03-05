import numpy as np
from model.utils.clip_around import clip, creat_mask
from model.lpgf_transformer.local_block_nc_new import LocalBlock
from model.lpgf_transformer.global_block_sim_low_high_new_5 import GlobalBlock
import torch
import torch.nn as nn
from model.loss_direct import local_loss, around_loss, global_loss, around_loss_4dr, time_loss
from model.layers.variable_embedding import VariableEmbedding
from model.layers.patch_wise_variable_aggregation import VariableAggregation_Linear as VariableAggregation


class LPGF_TF(nn.Module):
    def __init__(self, configs, return_h=False):
        super(LPGF_TF, self).__init__()
        self.return_h = return_h

        self.loss_mask = None
        self.w2 = None
        self.w1 = None
        self.configs = configs
        self.in_channel = configs.in_channel
        self.out_channel = configs.out_channel
        self.input_size = configs.input_size
        self.in_len = configs.in_len
        self.pred_len = configs.pred_len
        self.pretrain_pred_len = configs.pretrain_pred_len
        self.rebuild_start = configs.rebuild_start
        self.rebuild_len = self.in_len - self.rebuild_start - 1  # 重建前n-1帧
        self.local_len = self.in_len - self.rebuild_start + self.pretrain_pred_len
        self.pre_train = configs.pre_train
        self.win_patch = configs.win_patch
        self.around_patch = configs.around_patch
        self.patch_size = configs.patch_size
        self.position = configs.position

        self.vEmbd = VariableEmbedding(variables=self.in_channel,
                                       patch2=configs.patch_size ** 2,
                                       dim=configs.local_num_hidden,
                                       )
        self.vAggregation = VariableAggregation(blocks=configs.aggregation_blocks,
                                                variables=configs.in_channel,
                                                dim=configs.local_num_hidden,
                                                num_head=configs.aggregation_heads)

        self.local_proj = nn.Sequential(
            nn.Conv2d(in_channels=self.in_len * configs.local_num_hidden,
                      out_channels=configs.local_st_hidden, kernel_size=1),
            nn.LayerNorm([configs.local_st_hidden, configs.win_patch, configs.win_patch], elementwise_affine=True)
        )

        self.local_block = LocalBlock(configs)
        if not self.pre_train:
            self.global_block = GlobalBlock(configs)
            self.global_proj = nn.Sequential(
                nn.Conv2d(in_channels=self.in_len * configs.local_num_hidden,
                          out_channels=configs.global_num_hidden, kernel_size=1),
                nn.LayerNorm([configs.global_num_hidden, configs.win_patch, configs.win_patch], elementwise_affine=True)
            )

        self.clip_rate = configs.clip_rate
        self.mask_local, self.mask_around = creat_mask(self.win_patch, self.around_patch)
        self.getw()
        self.getmask(self.win_patch, self.around_patch)

    def getmask(self, wp, ap):

        mask_local, mask_around = creat_mask(wp, ap)
        m = []
        mask = torch.zeros([wp + ap * 2, wp + ap * 2, 2])
        for i in range(wp + ap * 2):
            for j in range(wp + ap * 2):
                mask[i, j, 0] = i
                mask[i, j, 1] = j

        ind = mask[mask_around, :]

        m.append([i for i in range(ind.shape[0]) if ind[i, 0] < ap or ind[i, 1] < ap])
        m.append([i for i in range(ind.shape[0]) if ind[i, 0] < ap or ind[i, 1] >= wp + ap])
        m.append([i for i in range(ind.shape[0]) if ind[i, 1] < ap or ind[i, 0] >= wp + ap])
        m.append([i for i in range(ind.shape[0]) if ind[i, 1] >= wp + ap or ind[i, 0] >= wp + ap])

        self.loss_mask = m

    def getw(self):
        T1 = self.in_len - self.rebuild_start + self.pretrain_pred_len
        T2 = self.in_len - self.rebuild_start - 1
        self.w1 = torch.Tensor(np.linspace(1, T1, T1)).cuda()  # 保留前n帧，预测后几帧
        self.w1 = self.w1 / self.w1.sum()

        self.w2 = torch.Tensor(np.linspace(1, T2, T2)).cuda()
        self.w2 = self.w2 / self.w2.sum()

    def forward(self, true):  # true is a list contains  surface, radar, era5, static

        if self.pre_train:
            local, around = clip(true, self.win_patch, self.around_patch,
                                 self.patch_size,
                                 self.mask_local, self.mask_around, self.clip_rate,
                                 )

            B, nindex, T, win_patch, win_patch, _ = local.shape
            T = T - self.pretrain_pred_len

            emb_local = local[:, :, 0:T, ...].reshape(B * nindex * T * win_patch * win_patch, -1)

            emb_local = self.vEmbd(emb_local)

            emb_local = self.vAggregation(emb_local)

            emb_local = emb_local.reshape(B, nindex, T, win_patch, win_patch, self.configs.local_num_hidden
                                          ).permute(0, 1, 2, 5, 3, 4
                                                    ).reshape(B * nindex, T * self.configs.local_num_hidden,
                                                              win_patch, win_patch)
            
            emb_local = self.local_proj(emb_local).reshape(B, nindex, self.configs.local_st_hidden, win_patch, win_patch
                                                      ).permute(0, 1, 3, 4, 2)

            # B, nindex, T, win_patch, win_patch, C
            # B, nindex, T, aroundpatch, C
            plocal, paround, direct = self.local_block(emb_local)

            return plocal, local[:, :, self.rebuild_start:self.rebuild_start + self.local_len, :, :,
                           :], \
                paround, around[:, :, self.rebuild_start:self.rebuild_start + self.rebuild_len, :, :], \
                direct
        else:
            if self.configs.train_together:
                B, _, _, H, W = true[0].shape
                T = self.in_len
                nwin = [H // (self.win_patch * self.patch_size),
                        W // (self.win_patch * self.patch_size)]

                data = true[0][:, 0:T, ...].reshape(
                    B, T, -1, nwin[0], self.win_patch, self.patch_size, nwin[1],
                    self.win_patch, self.patch_size
                ).permute(0, 3, 6, 1, 4, 7, 2, 5, 8
                        ).reshape(B * nwin[0] * nwin[1] * T * self.win_patch * self.win_patch, -1)

                emb_local = self.vEmbd(data)

                emb_local = self.vAggregation(emb_local)

                emb_local_all = emb_local.reshape(B, nwin[0] * nwin[1], T, self.win_patch, self.win_patch,
                                                self.configs.local_num_hidden).permute(
                    0, 1, 2, 5, 3, 4).reshape(B * nwin[0] * nwin[1], T * self.configs.local_num_hidden,
                                            self.win_patch, self.win_patch)

                emb_local = self.local_proj(emb_local_all)
                emb_local = emb_local.reshape(B, nwin[0] * nwin[1], self.configs.local_st_hidden,
                                            self.win_patch, self.win_patch
                                            ).permute(0, 1, 3, 4, 2)

                # B*nwinh*nwinw,dim, wp, wp(final h)
                h = self.local_block(emb_local)
            else:
                with torch.no_grad():
                    B, _, _, H, W = true[0].shape
                    T = self.in_len
                    nwin = [H // (self.win_patch * self.patch_size),
                            W // (self.win_patch * self.patch_size)]

                    data = true[0][:, 0:T, ...].reshape(
                        B, T, -1, nwin[0], self.win_patch, self.patch_size, nwin[1],
                        self.win_patch, self.patch_size
                    ).permute(0, 3, 6, 1, 4, 7, 2, 5, 8
                            ).reshape(B * nwin[0] * nwin[1] * T * self.win_patch * self.win_patch, -1)

                    emb_local = self.vEmbd(data)

                    emb_local = self.vAggregation(emb_local)

                    emb_local_all = emb_local.reshape(B, nwin[0] * nwin[1], T, self.win_patch, self.win_patch,
                                                    self.configs.local_num_hidden).permute(
                        0, 1, 2, 5, 3, 4).reshape(B * nwin[0] * nwin[1], T * self.configs.local_num_hidden,
                                                self.win_patch, self.win_patch)

                    emb_local = self.local_proj(emb_local_all)
                    emb_local = emb_local.reshape(B, nwin[0] * nwin[1], self.configs.local_st_hidden,
                                                self.win_patch, self.win_patch
                                                ).permute(0, 1, 3, 4, 2)

                    # B*nwinh*nwinw,dim, wp, wp(final h)
                    h = self.local_block(emb_local)

            # B,nwinh*win_patch,nwinw*win_patch, dim*ps*ps(for global block)
            h = h.reshape(B, nwin[0], nwin[1], self.win_patch, self.win_patch, self.configs.local_st_hidden)

            emb_x = emb_local_all.reshape(B * nwin[0] * nwin[1], T * self.configs.local_num_hidden,
                                          self.win_patch, self.win_patch)
            emb_x = self.global_proj(emb_x).reshape(B, nwin[0], nwin[1], self.configs.global_num_hidden,
                                              self.win_patch, self.win_patch).permute(0, 1, 2, 4, 5, 3)
                                              
            #pred = self.global_block(emb_x, torch.zeros(h.shape).cuda())
            pred = self.global_block(emb_x, h)
            # pred = self.global_block(h, h)
            
            #pred = self.global_block(emb_x, torch.zeros(h.shape).cuda())
            #pred = self.global_block(h)
            if self.return_h:
                return h.permute(0, 5, 1, 3, 2, 4).reshape(B, self.configs.local_st_hidden, nwin[0]*self.win_patch, nwin[1]*self.win_patch)
            else:
                return pred

    def pre_train_one_epoch(self, model, dataloader, optimizer, progress_bar):
        if not self.pre_train:
            raise ValueError('pre_train is False')
        Loss = 0
      
        for i, true in enumerate(dataloader):
            optimizer.zero_grad()
            true = true.cuda()
            plocal, local, paround, around, direct = model.forward(true)

            lloss = local_loss(plocal, local, self.w1)
            aloss = around_loss_4dr(paround, around, self.w2, direct, self.loss_mask)
            loss = lloss + aloss
            Loss = Loss + loss.item()
            loss.backward()

            # for name, param in self.named_parameters():
            #     if torch.isnan(param.grad).any():
            #         print(name)
            #         idex = param.grad[~torch.isnan(param.grad)]
            #         if len(idex) > 0:
            #             print(torch.max(abs(idex)))

            optimizer.step()
            progress_bar.set_postfix(
                {'loss': '{0:1.5f}'.format(loss), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            progress_bar.update(dataloader.batch_size)

        return Loss

    def train_one_epoch(self, model, dataloader, optimizer, progress_bar):
        if self.pre_train:
            raise ValueError('pre_train is True')
        Loss = 0
        for i, true in enumerate(dataloader):
            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()

            optimizer.zero_grad()
            pred = model.forward([true[0]])

            loss = global_loss(pred, true[1])
            #loss = time_loss(pred, true[1])
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

    def pre_train_vali(self, model, dataloader, progress_bar):  # 以loss为选择指标
        if not self.pre_train:
            raise ValueError('pre_train is False')
        Loss = 0
        for i, true in enumerate(dataloader):
            true = true.cuda()
            plocal, local, paround, around, direct = model.forward(true)

            lloss = local_loss(plocal, local, self.w1)
            aloss = around_loss_4dr(paround, around, self.w2, direct, self.loss_mask)
            loss = lloss + aloss
            Loss = Loss + loss.item()

            progress_bar.set_postfix(
                {'loss': '{0:1.5f}'.format(loss)})
            progress_bar.update(dataloader.batch_size)

        return Loss

    def train_vali(self, model, dataloader, progress_bar, metric, metrics, n_metrics):  # 以loss为选择指标,求所有leadtime的平均值
        if self.pre_train:
            raise ValueError('pre_train is True')
        Loss = 0
        for i, true in enumerate(dataloader):

            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()
            pred = model.forward([true[0]])

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
        if self.pre_train:
            raise ValueError('pre_train is True')
        for i, true in enumerate(dataloader):

            for k in range(2):
                true[k] = true[k].cuda()
            pred = model.forward([true[0]])

            # Denorm
            pred = dataloader.dataset.denorm(pred.cpu().detach())
            y = dataloader.dataset.denorm(true[1].cpu().detach())

            # B, C, H, W
            for j in range(self.configs.batch_size):
                for k in range(self.configs.pred_len):
                    metric.get_metrics(pred[j, k, ...], y[j, k, ...], metrics, n_metrics, k)
            progress_bar.update(dataloader.batch_size)
