import math
import numpy as np
import torch

def reserve_schedule_sampling_exp(itr, batch_size, args):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, args.in_len - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, args.pred_len - 1))
    true_token = (random_flip < eta)

    ones = np.ones((1,
                    1,
                    1))
    zeros = np.zeros((1,
                      1,
                      1))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.in_len+args.pred_len - 2):
            if j < args.in_len - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.pred_len - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.in_len+args.pred_len - 2,
                                  1,
                                  1,
                                  1))
    return torch.FloatTensor(real_input_flag)

def schedule_sampling(eta, itr, args):

    zeros = np.zeros((args.batch_size,
                      args.pred_len - 1,
                      1,
                      1,
                      1))

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (args.batch_size, args.pred_len - 1))
    true_token = (random_flip < eta)
    ones = np.ones((1, 1, 1))
    zeros = np.zeros((1, 1, 1))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.pred_len - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.pred_len - 1,
                                  1,
                                  1,
                                  1))
    return eta, torch.FloatTensor(real_input_flag)
