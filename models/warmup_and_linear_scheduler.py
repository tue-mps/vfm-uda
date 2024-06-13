# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


from logging import getLogger

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = getLogger()


class WarmupAndLinearScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, start_warmup_lr, warmup_iters, base_lr, final_lr, total_iters):
        warmup_lr_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)
        linear_lr_schedule = np.linspace(base_lr, final_lr, total_iters + 1)[warmup_iters:]
        self.lr_schedule = np.concatenate((warmup_lr_schedule, linear_lr_schedule))
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=-1, verbose=False)
        logger.info("Building LR scheduler done.")

    def get_lr(self):
        if self.last_epoch <= self.total_iters:
            i = self.last_epoch
        elif self.last_epoch < 0:
            i = 0
        else:
            i = self.last_epoch
            logger.warning("WarmupAndCosineScheduler: iter overflow: {}/{}".format(self.last_epoch,
                                                                                   self.total_iters))

        return [base_lr * self.lr_schedule[i] for base_lr in self.base_lrs]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.lr_schedule)), self.lr_schedule)
        plt.savefig('WarmupAndCosineScheduler.png')
        plt.close()
