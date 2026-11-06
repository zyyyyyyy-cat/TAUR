"""Explicit AMP-compatible gradient routing and reduction for TAUR steps."""
from collections import defaultdict
import torch
import torch.distributed as dist


def adapter_rank_gradients(rank_loss, adapter_parameters, scaler, weight):
    return torch.autograd.grad(scaler.scale(rank_loss * weight), adapter_parameters,
                               retain_graph=True, allow_unused=True)


@torch.no_grad()
def merge_adapter_gradients(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.add_(gradient)


@torch.no_grad()
def synchronize_gradients(model, bucket_mb=16):
    """Mean all local gradients once, after a DDP no_sync forward/backward.

    Synchronize before AMP unscale so every rank detects the same overflow.
    Globally unused parameters retain grad=None (no unwanted weight decay).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in parameters],
                        dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    groups = defaultdict(list)
    for parameter, is_used in zip(parameters, used.tolist()):
        if is_used:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            groups[(parameter.device, parameter.dtype)].append(parameter)
    for group in groups.values():
        bucket, size = [], 0
        for parameter in group:
            bucket.append(parameter)
            size += parameter.numel() * parameter.element_size()
            if size >= bucket_mb * 1024 ** 2:
                _reduce_bucket(bucket)
                bucket, size = [], 0
        if bucket:
            _reduce_bucket(bucket)


def _reduce_bucket(parameters):
    flat = torch.cat([p.grad.reshape(-1) for p in parameters])
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
        offset += count
