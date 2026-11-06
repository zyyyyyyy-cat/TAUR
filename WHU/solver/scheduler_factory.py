""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


class TAURCosineLRScheduler(CosineLRScheduler):
    """Retain UAD schedules and multiply its base schedule for Adapter groups."""
    def __init__(self, optimizer, base_lr, **kwargs):
        self.adapter_factors = [g.get('taur_lr_factor') for g in optimizer.param_groups]
        self.reference_group = next(i for i, g in enumerate(optimizer.param_groups)
                                    if g.get('taur_lr_factor') is None and g['lr'] == base_lr)
        super().__init__(optimizer, **kwargs)

    def _get_lr(self, t):
        values = super()._get_lr(t)
        base_value = values[self.reference_group]
        return [value if factor is None else factor * base_value
                for value, factor in zip(values, self.adapter_factors)]


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None

    scheduler_class = TAURCosineLRScheduler if cfg.TAUR.ENABLED else CosineLRScheduler
    extra = {'base_lr': cfg.SOLVER.BASE_LR} if cfg.TAUR.ENABLED else {}
    lr_scheduler = scheduler_class(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
            **extra,
        )

    return lr_scheduler
