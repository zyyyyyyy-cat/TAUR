"""Two-process Gloo check of routing, zero relations, AMP and PKM sharding.

Run: python tests/check_taur_ddp.py (no CUDA or dataset required).
"""
import copy
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.backbones.vit_pytorch import TransReID
from processor.taur_grad import (adapter_rank_gradients, merge_adapter_gradients, synchronize_gradients,
                                direction_gradient_diagnostics, gradient_contribution, step_with_scaler)
from loss.taur_rank_loss import TAURRankingLoss
from datasets.sampler_camera import CameraPKMSampler
from amp_compat import cpu_scaler


def make_model():
    model = TransReID(img_size=(32, 16), stride_size=16, embed_dim=24,
                     num_heads=3, depth=12, num_classes=3, drop_path_rate=0)
    model.enable_thermal_adapters([6, 9], 8)
    return model


def local_metadata(rank, invalid=False):
    pid = torch.tensor([0, 0, 1, 1] * 3) + rank * 2
    cam = torch.tensor([0, 1, 0, 1] * 3)
    if invalid:
        cam.zero_()
    mod = torch.tensor([1] * 4 + [2] * 4 + [3] * 4)
    return pid, cam, mod, torch.arange(12) + rank * 12


def reference_gradients(reference, inputs, invalid, weight):
    # Average of LOCAL conditional objectives, not a global-gallery loss.
    contributions = []
    for rank in range(2):
        reference.zero_grad(set_to_none=True)
        meta = local_metadata(rank, invalid and rank == 1)
        f = reference(inputs[rank], modal_label=meta[2])
        base = f.square().mean()
        extra, _, _ = TAURRankingLoss()(f, *meta)
        adapters = tuple(reference.thermal_adapters.parameters())
        grads = adapter_rank_gradients(extra, adapters, cpu_scaler(enabled=False), weight)
        base.backward()
        merge_adapter_gradients(adapters, grads)
        contributions.append([None if p.grad is None else p.grad.clone() for p in reference.parameters()])
    return [None if a is None and b is None else
            ((torch.zeros_like(b) if a is None else a) + (torch.zeros_like(a) if b is None else b)) / 2
            for a, b in zip(*contributions)]


def worker(rank, port):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:' + str(port), rank=rank, world_size=2)
    torch.manual_seed(5)
    model = make_model()
    reference = copy.deepcopy(model)
    ddp = DDP(model, find_unused_parameters=True)
    generator = torch.Generator().manual_seed(10)
    inputs = [torch.randn(12, 3, 32, 16, generator=generator) for _ in range(2)]
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
    # CPU scaling exercises scaled-gradient merging; CUDA FP16 remains a server check.
    scaler = cpu_scaler(init_scale=128)
    for step in range(5):
        weight = 0 if step == 0 else 0.1
        invalid = step == 2
        reference.load_state_dict(model.state_dict())
        expected = reference_gradients(reference, inputs, invalid, weight)
        original = copy.deepcopy(model.state_dict())
        optimizer.zero_grad(set_to_none=True)
        meta = local_metadata(rank, invalid and rank == 1)
        context = ddp.no_sync() if weight else __import__('contextlib').nullcontext()
        with context:
            f = ddp(inputs[rank], modal_label=meta[2])
            base = f.square().mean()
            extra, directions, _ = TAURRankingLoss()(f, *meta)
            adapters = tuple(model.thermal_adapters.parameters())
            if weight:
                report = direction_gradient_diagnostics(directions, adapters, scaler, weight, ddp=True)
                assert all(torch.isfinite(torch.tensor(v['mean_gradient_norm'])) for v in report.values())
            grads = adapter_rank_gradients(extra, adapters, scaler, weight) if weight else None
            scaler.scale(base).backward()
            if grads is not None:
                contribution = gradient_contribution(adapters, grads, scaler, ddp=True)
                assert contribution['uad_adapter_norm'] >= 0
                merge_adapter_gradients(adapters, grads)
        if step == 4 and rank == 1:
            next(model.parameters()).grad.fill_(float('inf'))
        if weight:
            synchronize_gradients(ddp, bucket_mb=0.001)
        scaler.unscale_(optimizer)
        for parameter, gradient in zip(model.parameters(), expected):
            if gradient is None:
                assert parameter.grad is None
            elif step != 4:
                torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)
        updated, before, after = step_with_scaler(optimizer, scaler)
        if step == 4:
            assert not updated and after < before
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        else:
            assert updated
        for p in model.parameters():
            peer = p.detach().clone()
            dist.broadcast(peer, src=0)
            torch.testing.assert_close(p, peer, atol=0, rtol=0)

    data = {m: [('{}_{}_{}'.format(m, pid, i), pid, i % 2, mid)
                for pid in range(8) for i in range(8)]
            for mid, m in enumerate(['RGB', 'IR', 'Thermal'], 1)}
    sampler = CameraPKMSampler(data, 16, 4, list(data), rank=rank, world_size=2)
    sampled = list(sampler)
    gathered = [None, None]
    dist.all_gather_object(gathered, sampled)
    assert all(len(part) == len(sampler) for part in gathered)
    for step in range(sampler.steps):
        ids = [{data['RGB'][t[0]][1] for t in part[step * 8:(step + 1) * 8]} for part in gathered]
        assert len(ids[0]) == len(ids[1]) == 2 and ids[0].isdisjoint(ids[1])
    if rank == 0:
        print('PASS: 2-process DDP, base-to-routed transition, AMP scaling and diagnostics, '
              'one-rank overflow with all-rank skipping, zero-direction rank, reference mean gradients, '
              'unused head, equal parameters, camera PKM shards.')
    dist.destroy_process_group()


if __name__ == '__main__':
    # Windows Gloo should bind loopback, not a disconnected host interface.
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo' if sys.platform != 'win32' else 'loopback_0')
    if not os.environ['GLOO_SOCKET_IFNAME']:
        del os.environ['GLOO_SOCKET_IFNAME']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
