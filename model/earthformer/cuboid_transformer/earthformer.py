from torch import nn
from omegaconf import OmegaConf
from model.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
import os
import torch

from model.loss import global_loss


class EarthFormer(nn.Module):
    def __init__(self, configs):
        super(EarthFormer, self).__init__()
        fil = os.path.dirname(__file__)
        oc_file = os.path.join(fil, "earthformer_sevir_v1.yaml")
        model_cfg = OmegaConf.to_object(OmegaConf.load(open(oc_file, "r")).model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.configs = configs
        self.model = CuboidTransformerModel(
            input_shape=[configs.in_len, configs.input_size[0], configs.input_size[1], configs.in_channel],
            target_shape=[configs.pred_len, configs.input_size[0], configs.input_size[1], configs.in_channel],
            base_units=model_cfg["base_units"],
            # block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

    def forward(self, x):
        x = x[0][:, :self.configs.in_len, ...]
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x).permute(0, 1, 4, 2, 3)
        return x

    def train_one_epoch(self, model, dataloader, optimizer, progress_bar):
        Loss = 0
        for i, true in enumerate(dataloader):
            Loss2 = 0
            for k in range(2):
                true[k] = true[k].cuda()

            optimizer.zero_grad()
            pred = model.forward([true[0]])

            loss = global_loss(pred[:, :, 0, :, :], true[1][:, :, 0, :, :])
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
            pred = model.forward([true[0]])

            loss = global_loss(pred[:, :, 0, :, :], true[1][:, :, 0, :, :])
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
            pred = model.forward([true[0]])

            # Denorm
            pred = dataloader.dataset.denorm(pred.cpu().detach())
            y = dataloader.dataset.denorm(true[1].cpu().detach())

            # B, C, H, W
            for j in range(self.configs.batch_size):
                for k in range(self.configs.pred_len):
                    metric.get_metrics(pred[j, k, ...], y[j, k, ...], metrics, n_metrics, k)
            progress_bar.update(dataloader.batch_size)
