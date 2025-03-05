import random
import torch
from torch import nn
from model.loss_direct import global_loss
from model.modules import PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN, K2M


class PhyDNet(nn.Module):
    r"""PhyDNet Model

    Implementation of `Disentangling Physical Dynamics from Unknown Factors for
    Unsupervised Video Prediction <https://arxiv.org/abs/2003.01460>`_.

    """

    def __init__(self, configs):
        super(PhyDNet, self).__init__()
        self.configs = configs
        self.in_len = configs.in_len
        self.out_len = configs.pred_len
        H, W = configs.input_size
        C = configs.in_channel
        self.n_layers_p = 1
        self.n_layers_c = configs.n_layers_c
        self.kernal_size_c = configs.kernal_size_c
        self.input_dim = configs.input_dim
       
        self.hidden_dims = configs.hidden_dim
        patch_size = configs.patch_size
        input_shape = (H // patch_size, W // patch_size)

        self.phycell = PhyCell(input_shape=input_shape, input_dim=self.input_dim, F_hidden_dims=[49],
                               n_layers=self.n_layers_p, kernel_size=(7,7))
        self.convcell = PhyD_ConvLSTM(input_shape=input_shape, input_dim=self.input_dim, hidden_dims=self.hidden_dims,
                                      n_layers=self.n_layers_c, kernel_size=(self.kernal_size_c, self.kernal_size_c))
        self.encoder = PhyD_EncoderRNN(self.phycell, self.convcell,
                                       in_channel=C, patch_size=patch_size)
        self.k2m = K2M([7,7])

        self.criterion = nn.MSELoss()
        self.constraints = self._get_constraints()
    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 


    def forward(self, inputs, return_loss=False):
        input_tensor = inputs[0]
        target_tensor = inputs[1]
      
        loss = 0
        for ei in range(self.in_len - 1):
            encoder_output, encoder_hidden, output_image, _, _  = \
                self.encoder(input_tensor[:,ei,:,:,:], (ei==0))
            output_image = output_image[:, 0:self.configs.out_channel, :, :]
            if return_loss:
                loss += self.criterion(output_image, input_tensor[:,ei+1,:,:,:])

        decoder_input = input_tensor[:,-1,:,:,:]
        predictions = []

        for di in range(self.out_len):
            _, _, output_image, _, _ = self.encoder(decoder_input, False, False)
            decoder_input = output_image
            output_image = output_image[:, 0:self.configs.out_channel, :, :]
            predictions.append(output_image)
            if return_loss:
                loss += self.criterion(output_image, target_tensor[:,di,:,:,:])

        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]
            m = self.k2m(filters.double()).float()
            if return_loss:
                loss += self.criterion(m, self.constraints.cuda())
        if return_loss:
            return torch.stack(predictions, dim=1), loss
        else:
            return torch.stack(predictions, dim=1)
        
    def train_one_epoch(self, model, dataloader, optimizer, progress_bar):
        Loss = 0
        for i, true in enumerate(dataloader):
            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()

       

            optimizer.zero_grad()
            pred, loss = model.forward([true[0], true[1]], return_loss= True)
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


        
