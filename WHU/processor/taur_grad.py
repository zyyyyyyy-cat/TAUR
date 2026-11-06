"""Explicit AMP-compatible gradient routing and reduction for TAUR steps."""
from collections import defaultdict
import torch
import torch.distributed as dist


def adapter_rank_gradients(rank_loss, adapter_parameters, scaler, weight):
    return torch.autograd.grad(scaler.scale(rank_loss * weight), adapter_parameters,
                               retain_graph=True, allow_unused=True)


def gradient_vector(parameters, gradients, scale=1.0):
    """Unscale returned gradients in FP32 without touching parameter .grad."""
    return torch.cat([(torch.zeros_like(p, dtype=torch.float32) if g is None
                       else g.detach().float() / scale).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def direction_gradient_diagnostics(losses, parameters, scaler, weight, ddp=False):
    report = {}
    for direction, value in losses.items():
        grads = adapter_rank_gradients(value, parameters, scaler, 0.25 * weight)
        vector = gradient_vector(parameters, grads, scaler.get_scale())
        local_norm = vector.norm().item()
        if ddp:
            dist.all_reduce(vector)
            vector.div_(dist.get_world_size())
        report[direction] = {'local_norm': local_norm, 'mean_gradient_norm': vector.norm().item()}
    return report


@torch.no_grad()
def gradient_contribution(parameters, rank_gradients, scaler, ddp=False):
    base = gradient_vector(parameters, [p.grad for p in parameters], scaler.get_scale())
    extra = gradient_vector(parameters, rank_gradients, scaler.get_scale())
    if ddp:
        dist.all_reduce(base)
        dist.all_reduce(extra)
        base.div_(dist.get_world_size())
        extra.div_(dist.get_world_size())
    base_norm, rank_norm = base.norm().item(), extra.norm().item()
    return {'uad_adapter_norm': base_norm, 'rank_adapter_norm': rank_norm,
            'rank_to_uad_ratio': rank_norm / base_norm if base_norm > 0 else None,
            'cosine': (torch.dot(base, extra) / (base.norm() * extra.norm())).item()
            if base_norm > 0 and rank_norm > 0 else None}


def step_with_scaler(optimizer, scaler):
    """Public GradScaler API: a backoff means the optimizer step was skipped."""
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after = scaler.get_scale()
    return after >= before, before, after


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
