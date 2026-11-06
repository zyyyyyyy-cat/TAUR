"""Regression checks for the training-log audit corrections."""
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch import nn
from amp_compat import FixedCPUScaler, cpu_scaler
from processor.taur_grad import (direction_gradient_diagnostics, gradient_vector,
                                gradient_contribution, step_with_scaler)
from datasets.sampler_camera import CameraPKMSampler
from model.make_model import load_reid_state, read_reid_checkpoint
from loss.wrt_loss import softmax_weights, TripletLoss_WRT
from config import cfg


def test_fp16_diagnostic_recovers_underflow_without_modifying_grad():
    p = nn.Parameter(torch.tensor([1.0]))
    # Half backward bridge reproduces the dtype path without CPU Half GEMM.
    loss = p.half().float().sum() * 1e-10
    old = torch.autograd.grad(loss * 0.25, p, retain_graph=True)[0]
    assert old.item() == 0
    report = direction_gradient_diagnostics({'T2T': loss}, (p,),
                                             FixedCPUScaler(init_scale=65536), 1.0)
    assert report['T2T']['local_norm'] > 0
    assert report['T2T']['mean_gradient_norm'] == report['T2T']['local_norm']
    assert p.grad is None


def test_gradient_report_and_none_parameters():
    a, b = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(1))
    a.grad = torch.tensor([384., 512.])
    b.grad = None
    rank = (torch.tensor([128., 0.]), None)
    report = gradient_contribution((a, b), rank, FixedCPUScaler(init_scale=128))
    assert report['uad_adapter_norm'] == 5
    assert report['rank_adapter_norm'] == 1
    assert report['rank_to_uad_ratio'] == 0.2
    assert report['cosine'] == pytest.approx(0.6)
    torch.testing.assert_close(a.grad, torch.tensor([384., 512.]))
    assert b.grad is None


@pytest.mark.parametrize('native', [False, True])
def test_scaler_step_counts_and_overflow(native):
    p = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([p], lr=0.1)
    scaler = cpu_scaler(init_scale=128) if native else FixedCPUScaler(init_scale=128)
    for overflow in (False, True, False):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(p.sum()).backward()
        if overflow:
            p.grad.fill_(float('inf'))
        scaler.unscale_(optimizer)
        before = p.detach().clone()
        updated, first, last = step_with_scaler(optimizer, scaler)
        assert updated != overflow
        if overflow:
            assert first > last and torch.equal(before, p)
        else:
            torch.testing.assert_close(p, before - 0.1)


def test_cpu_scaler_compatibility_without_torch_amp_gradscaler(monkeypatch):
    monkeypatch.delattr(torch.amp, 'GradScaler', raising=False)
    assert isinstance(cpu_scaler(), FixedCPUScaler)


def data_for_counts(counts, scarce=False):
    return {m: [(str((m, p, i)), p, (int(i == count - 1) if scarce else i % 2), mid)
                for p, count in enumerate(counts) for i in range(count)]
            for mid, m in enumerate(('RGB', 'IR', 'Thermal'), 1)}


def test_sampler_full_coverage_consumes_epoch_pool():
    data = data_for_counts([100] * 16)
    sampler = CameraPKMSampler(data, 64, 4, list(data), seed=1234)
    sampler.set_epoch(1)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 1600
    for m in range(3):
        assert len({row[m] for row in indices}) == 1600
    assert sampler.last_stats['unique_images'] == {m: 1600 for m in data}


def test_sampler_scarce_camera_not_recycled():
    data = data_for_counts([100], scarce=True)
    sampler = CameraPKMSampler(data, 4, 4, list(data))
    indices = list(sampler)
    assert len({row[0] for row in indices}) == 100
    groups = [indices[i:i + 4] for i in range(0, 100, 4)]
    assert sum(len({data['RGB'][row[0]][2] for row in group}) == 2 for group in groups) == 1


def test_sampler_randomized_budget_and_four_rank_shards():
    rng = random.Random(61)
    for _ in range(40):
        counts = [rng.randint(1, 20) * 4 for _ in range(16)]
        data = data_for_counts(counts)
        single = CameraPKMSampler(data, 32, 4, list(data), seed=9)
        full = list(single)
        parts = [list(CameraPKMSampler(data, 32, 4, list(data), seed=9, rank=r, world_size=4))
                 for r in range(4)]
        assert len(full) == len(single) == len(parts[0]) * 4
        assert len(set(full)) == len(full)  # balanced modalities: no repeated raw indices
        for step in range(single.steps):
            combined = sum([part[step * 8:(step + 1) * 8] for part in parts], [])
            assert combined == full[step * 32:(step + 1) * 32]
            assert len({data['RGB'][row[0]][1] for row in combined}) == 8


def test_sampler_undersized_modality_balances_repeats():
    data = data_for_counts([20])
    data['IR'] = data['IR'][:3]
    samples = list(CameraPKMSampler(data, 4, 4, list(data)))
    counts = [sum(row[1] == i for row in samples) for i in range(3)]
    assert max(counts) - min(counts) <= 1
    assert len({row[0] for row in samples}) == 20


def checkpoint_model():
    model = nn.Module()
    model.base = nn.Linear(3, 3)
    model.base.thermal_adapters = nn.ModuleDict({'6': nn.Linear(3, 3), '9': nn.Linear(3, 3)})
    model.bottleneck = nn.BatchNorm1d(3)
    model.classifier = nn.Linear(3, 2)
    return model


@pytest.mark.parametrize('defect', ['missing_shared', 'missing_bn', 'partial_adapter', 'shape', 'unexpected'])
def test_checkpoint_rejection_is_atomic(defect):
    model = checkpoint_model()
    state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    if defect == 'missing_shared':
        del state['base.weight']
    elif defect == 'missing_bn':
        del state['bottleneck.running_mean']
    elif defect == 'partial_adapter':
        del state['base.thermal_adapters.6.weight']
    elif defect == 'shape':
        state['base.thermal_adapters.9.weight'] = torch.zeros(2, 2)
    else:
        state['unexpected.weight'] = torch.ones(2)
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        load_reid_state(model, state)
    for name, value in model.state_dict().items():
        assert torch.equal(original[name], value)


def test_checkpoint_uad_initialization_and_container_prefix(tmp_path):
    model = checkpoint_model()
    state = {k: torch.ones_like(v) for k, v in model.state_dict().items()
             if 'thermal_adapters.' not in k}
    initial = copy.deepcopy(model.base.thermal_adapters.state_dict())
    load_reid_state(model, state, finetune=True)
    assert torch.equal(model.base.weight, torch.ones_like(model.base.weight))
    for name, value in initial.items():
        assert torch.equal(value, model.base.thermal_adapters.state_dict()[name])
    with pytest.raises(ValueError, match='Adapter mismatch'):
        load_reid_state(model, state)
    wrapped = {'model': {'module.' + k: v for k, v in model.state_dict().items()}}
    torch.save(wrapped, tmp_path / 'full.pth')
    load_reid_state(model, read_reid_checkpoint(tmp_path / 'full.pth'))


def test_stable_wrt_large_negative_distances_and_empty_mask():
    logits = torch.tensor([[0., -20., -25.], [0., -2000., -2005.], [0., 0., 0.]], requires_grad=True)
    mask = torch.tensor([[0., 1., 1.], [0., 1., 1.], [0., 0., 0.]])
    weights = softmax_weights(logits, mask, stable=True)
    torch.testing.assert_close(weights.sum(1), torch.tensor([1., 1., 0.]))
    torch.testing.assert_close(weights[0, 1:], torch.softmax(torch.tensor([-20., -25.]), 0))
    torch.testing.assert_close(weights[0], weights[1])
    (weights * torch.tensor([0., 20., 25.])).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[2].count_nonzero() == 0
    assert softmax_weights(logits.detach()[:1], mask[:1]).sum() < 0.01


def test_stable_wrt_matches_direct_mathematical_loss_and_gradient():
    torch.manual_seed(11)
    features = (torch.randn(8, 5) * 10).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    actual = TripletLoss_WRT(stable=True)(features, labels)[0]
    from loss.wrt_loss import pdist_torch
    distances = pdist_torch(features, features)
    reference = []
    for i in range(8):
        positives = distances[i, labels == labels[i]]
        negatives = distances[i, labels != labels[i]]
        ap = (positives * torch.softmax(positives, 0)).sum()
        an = (negatives * torch.softmax(-negatives, 0)).sum()
        reference.append(torch.nn.functional.softplus(ap - an))
    reference = torch.stack(reference).mean()
    torch.testing.assert_close(actual, reference)
    a = torch.autograd.grad(actual, features, retain_graph=True)[0]
    b = torch.autograd.grad(reference, features)[0]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_legacy_and_matched_control_configuration():
    root = Path(__file__).resolve().parents[1]
    old, taur, control = cfg.clone(), cfg.clone(), cfg.clone()
    old.merge_from_file(str(root / 'configs/UAD.yml'))
    taur.merge_from_file(str(root / 'configs/TAUR.yml'))
    control.merge_from_file(str(root / 'configs/UAD_TAUR_control.yml'))
    assert not old.MODEL.WRT_STABLE and not old.DATALOADER.CAMERA_PRIORITY
    assert taur.MODEL.WRT_STABLE and control.MODEL.WRT_STABLE
    assert taur.DATALOADER.CAMERA_PRIORITY and control.DATALOADER.CAMERA_PRIORITY
    assert taur.TAUR.ENABLED and not control.TAUR.ENABLED


def test_real_trainer_six_epoch_transition_validation_and_inference(caplog):
    """Exercise the actual trainer, real UAD losses, sampler and eval entrypoints."""
    from model.backbones.vit_pytorch import TransReID
    from datasets.bases import ImageDataset
    from datasets.make_dataloader import taur_train_collate_fn
    from processor.processor import do_train, do_inference
    from solver.make_optimizer import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from loss.make_loss import make_loss
    from torch.utils.data import DataLoader
    import datasets.bases as bases
    from unittest.mock import patch
    import logging

    class TinyUAD(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = TransReID(img_size=(32, 16), stride_size=16, embed_dim=24,
                                  num_heads=3, depth=12, num_classes=3, drop_path_rate=0)
            self.base.enable_thermal_adapters([6, 9], 8)
            self.bottleneck = nn.BatchNorm1d(24)
            self.classifier = nn.Linear(24, 2)
            self.in_planes, self.num_classes = 24, 2

        def adapter_parameters(self):
            return self.base.thermal_adapters.parameters()

        def forward(self, images, label=None, camids=None, mode=0, modality_ids=None):
            x = torch.cat(images) if mode == 0 else images
            f = self.base(x, modal_label=modality_ids)
            bn = self.bottleneck(f)
            return (self.classifier(bn), f, bn) if mode == 0 else f

    c = cfg.clone()
    c.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/TAUR.yml'))
    c.MODEL.DEVICE = 'cpu'
    c.SOLVER.MAX_EPOCHS, c.SOLVER.IMS_PER_BATCH = 6, 8
    c.SOLVER.LOG_PERIOD, c.SOLVER.EVAL_PERIOD, c.SOLVER.CHECKPOINT_PERIOD = 1, 6, 999
    c.TAUR.LOG_DIRECTION_GRADS = True
    torch.manual_seed(49)
    images = torch.randn(8, 3, 32, 16)
    data = data_for_counts([4, 4])
    train = ImageDataset(data, list(data), return_metadata=True)
    sampler = CameraPKMSampler(data, 8, 4, list(data))
    loader = DataLoader(train, sampler=sampler, batch_size=8, collate_fn=taur_train_collate_fn)
    vals = []
    for mod in (1, 2, 3):
        vals.append([(torch.randn(5, 3, 32, 16), (0, 0, 0, 1, 1),
                      (0, 1, 1, 1, 1), torch.tensor([0, 1, 1, 1, 1]),
                      (mod,) * 5, ('q', 'g0', 'g1', 'g2', 'g3'))])
    model = TinyUAD()
    before = copy.deepcopy(model.state_dict())
    loss_fn, center = make_loss(c, 2)
    optimizer, center_optimizer = make_optimizer(c, model, center)
    scheduler = create_scheduler(c, optimizer)
    caplog.set_level(logging.INFO)
    with patch.object(bases, 'read_image', side_effect=lambda path: images[abs(hash(path)) % 8]):
        do_train(c, model, center, loader, vals, optimizer, center_optimizer,
                 scheduler, loss_fn, [1, 1, 1], 0)
    rank1, mean_ap = do_inference(c, model, vals, [1, 1, 1])
    assert 0 <= rank1 <= 1 and 0 <= mean_ap <= 1
    assert not torch.equal(before['base.thermal_adapters.6.up.weight'],
                           model.state_dict()['base.thermal_adapters.6.up.weight'])
    assert model.base.fc.weight.grad is None
    assert 'AMP-safe direction gradients' in caplog.text
    assert 'uad_adapter_norm' in caplog.text and 'rank_adapter_norm' in caplog.text
    assert 'cumulative attempts=6 updates=6 skips=0' in caplog.text
    assert 'Validation Results - Epoch: 6' in caplog.text
