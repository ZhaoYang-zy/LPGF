import random
import torch


def creat_mask(win_patch, around_patch):
    mask_tensor = torch.ones((win_patch + around_patch * 2, win_patch + around_patch * 2))
    mask_tensor[1:1 + win_patch, 1:1 + win_patch] = 0
    mask_local = mask_tensor == 0
    mask_around = mask_tensor == 1
    return mask_local, mask_around


def clip(true, win_patch, around_patch, patch_size, mask_local, mask_around, rate):
    # random clip

    B, T, C, H, W = true.shape
    H = H // patch_size
    W = W // patch_size

    nwin = [H // (win_patch + around_patch * 2), W // (win_patch + around_patch * 2)]
    Hnew = nwin[0] * (win_patch + around_patch * 2)
    Wnew = nwin[1] * (win_patch + around_patch * 2)
    Hbegin = (H - Hnew) // 2
    Wbegin = (W - Wnew) // 2

    true = true.reshape(B, T, C, H, patch_size, W, patch_size). \
        permute(0, 1, 2, 4, 6, 3, 5).reshape(B, T, C * patch_size ** 2, H, W)

    C = C * patch_size ** 2

    data = true[:, :, :, Hbegin:Hbegin + Hnew, Wbegin:Wbegin + Wnew].reshape(B, T, C, nwin[0],
                                                                             win_patch + around_patch * 2,
                                                                             nwin[1], win_patch + around_patch * 2)
    data = data.permute(0, 3, 5, 1, 2, 4, 6)  # B,nwin0,nwin1,T,C,wp+ap,wp+ap

    data = torch.reshape(data, (B, nwin[0] * nwin[1], T, C, win_patch + around_patch * 2,
                                win_patch + around_patch * 2,))

    nindex = int(nwin[0] * nwin[1] * rate)

    index = torch.LongTensor(random.sample(range(nwin[0] * nwin[1]), nindex))
    data = data[:, index, ...]

    around = data[..., mask_around].permute(0, 1, 2, 4, 3)  # B,nindex,T,aroundpatch,C

    # B, nindex, T, win_patch, win_patch, C
    local = data[..., mask_local].reshape(B, nindex, T, C, win_patch, win_patch).permute(0, 1, 2, 4, 5, 3)

    return local, around
