import argparse


def creat_parser():
    parser = argparse.ArgumentParser(description='')
    
    # train
    parser.add_argument('--pre_train', type=bool, default=False)
    parser.add_argument('--continue_train', type=bool, default=True)
    parser.add_argument('--exp_name', type=str, default='sevir')
    parser.add_argument('--exp_describ', type=str, default='')
    parser.add_argument('--model', type=str, default='phydnet')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--out_channel', type=int, default=1)
    parser.add_argument('--result_root', type=str, default='result')
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--lr', default=0.0002)
    parser.add_argument('--step_size', default=2, type=int, help='for scheduler to step')
    parser.add_argument('--scheduler_gamma', default=0.9, type=float)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', default=24)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_thresh', type=int, default=0)
    parser.add_argument('--at_drop', type=float, default=0.0)
    parser.add_argument('--pj_drop', type=float, default=0.0)
    parser.add_argument('--position', type=bool, default=False)

    parser.add_argument('--vali_metrics',
                        default=['csi_30', 'csi_45', 'pod_20', 'far_20', 'psnr', 'ssim',
                                 'mse'])
    parser.add_argument('--test_metrics',
                        default=['csi_30', 'csi_45', 'pod_20', 'far_20', 'psnr', 'ssim',
                                 'mse'])

    # data

    parser.add_argument('--input_size', type=list, default=[256, 256])

    parser.add_argument('--in_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)

    # variableAggregation
    parser.add_argument('--aggregation_blocks', type=int, default=3)
    parser.add_argument('--aggregation_heads', type=int, default=3)

    # rnn
    parser.add_argument('--rnn_num_layers', type=int, default=4)
    parser.add_argument('--rnn_num_hidden', type=int, default=128, help='')
    parser.add_argument('--rnn_filter_size', type=int, default=5)
    parser.add_argument('--rnn_stride', type=int, default=1)
    parser.add_argument('--rnn_layer_norm', type=bool, default=True)
    parser.add_argument('--reverse_scheduled_sampling', type=bool, default=False)
    parser.add_argument('--decouple_beta', type=float, default=0.1)

    # local
    parser.add_argument('--pretrain_pred_len', type=int, default=4)
    parser.add_argument('--local_blocks', type=int, default=3)
    parser.add_argument('--local_channel_blocks', type=int, default=1)
    parser.add_argument('--local_spatio_blocks', type=int, default=2)
    parser.add_argument('--local_channel_num_head', type=int, default=4)
    parser.add_argument('--local_spatio_num_head', type=int, default=4)

    parser.add_argument('--paround_blocks', type=int, default=3)
    parser.add_argument('--paround_num_head', type=int, default=4)

    parser.add_argument('--plocal_c_blocks', type=int, default=3)
    parser.add_argument('--plocal_c_num_head', type=int, default=4)
    parser.add_argument('--plocal_s_blocks', type=int, default=3)
    parser.add_argument('--plocal_s_num_head', type=int, default=4)

    parser.add_argument('--direct_bias', type=float, default=0.3)

    parser.add_argument('--local_num_hidden', type=int, default=256)
    parser.add_argument('--local_st_hidden', type=int, default=256)

    parser.add_argument('--rebuild_start', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--win_patch', type=int, default=4, help='local patch nh nw')
    parser.add_argument('--around_patch', type=int, default=1)
    parser.add_argument('--clip_rate', type=float, default=1)

    # global
    parser.add_argument('--train_together', type=bool, default=True, help='')
    parser.add_argument('--train_direct', type=bool, default=False, help='')
    parser.add_argument('--global_num_hidden', type=int, default=256, help='')
    parser.add_argument('--global_blocks', type=int, default=6)
    parser.add_argument('--channel_blocks', type=int, default=1)
    parser.add_argument('--spatio_blocks', type=int, default=8)
    parser.add_argument('--channel_num_head', type=int, default=4)
    parser.add_argument('--spatio_num_head', type=int, default=4)
    parser.add_argument('--shift_stride', type=int, default=2)

    # visible
    parser.add_argument('--flush_secs', type=int, default=120)

    return parser
