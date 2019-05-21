from bisect import bisect_right

from torch.optim.lr_scheduler import MultiStepLR


class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super(MultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)
        self.milestones = milestones
        self.last_epoch = 0

    def get_lr(self):
        lr = self.base_lrs
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]
        return [l * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for l in lr]
