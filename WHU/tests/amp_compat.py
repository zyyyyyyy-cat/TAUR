"""Test-only CPU scaling fallback for Torch 2.2 (not a CUDA AMP replacement)."""
import torch


class FixedCPUScaler:
    def __init__(self, enabled=True, init_scale=128.0):
        self.value = float(init_scale) if enabled else 1.0
        self.enabled, self.finite = enabled, True

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, optimizer):
        self.finite = True
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.value)
                    self.finite &= bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        if self.finite:
            return optimizer.step()

    def update(self):
        if self.enabled and not self.finite:
            self.value *= 0.5


def cpu_scaler(enabled=True, init_scale=128.0):
    factory = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if factory is not None:
        return factory('cpu', enabled=enabled, init_scale=init_scale)
    return FixedCPUScaler(enabled, init_scale)
