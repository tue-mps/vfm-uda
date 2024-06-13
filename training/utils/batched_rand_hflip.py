import torch
from torchvision.transforms import functional as FV


def batched_rand_hflip(x, y):
    b, c, h, w = x.shape

    rv_x, rv_y = [] , []
    for i in range(b):
        if torch.rand(1) < 0.5:
            rv_x += [FV.hflip(x[i])]
            rv_y += [FV.hflip(y[i])]
        else:
            rv_x += [x[i]]
            rv_y += [y[i]]

    return torch.stack(rv_x), torch.stack(rv_y)