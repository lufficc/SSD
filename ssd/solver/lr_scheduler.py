from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
