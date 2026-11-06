"""Data-free behavioral checks. Run from WHU: python -m pytest -q tests."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch import nn
from config import cfg
from model.backbones.thermal_adapter import ThermalAdapter
from model.backbones.vit_pytorch import TransReID
from model.make_model import build_transformer
from loss.taur_rank_loss import TAURRankingLoss, rank_weight
from processor.taur_grad import adapter_rank_gradients, merge_adapter_gradients
from processor.taur_grad import synchronize_gradients
from processor.processor import PCA, GPD, PrototypeBank
from datasets.sampler_camera import CameraPKMSampler
from datasets.bases import ImageDataset
from datasets.make_dataloader import taur_train_collate_fn
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler

torch.set_num_threads(2)


def tiny_backbone():
    return TransReID(img_size=(32, 16), stride_size=16, embed_dim=24,
                    num_heads=3, depth=12, num_classes=3, drop_path_rate=0)


def metadata():
    pid = torch.tensor([0, 0, 1, 1] * 3)
    cam = torch.tensor([0, 1, 0, 1] * 3)
    mod = torch.tensor([1] * 4 + [2] * 4 + [3] * 4)
    return pid, cam, mod, torch.arange(12)


def brute_rank(features, pid, cam, mod, sid):
    s = nn.functional.normalize(features.float(), dim=1)
    s = s @ s.t()
    result = {}
    for name, qm, gm in [('T2T', 3, 3), ('T2ALL', 3, None), ('R2T', 1, 3), ('N2T', 2, 3)]:
        queries = []
        for i in range(len(pid)):
            if mod[i] != qm:
                continue
            eligible = [j for j in range(len(pid)) if cam[i] != cam[j] and sid[i] != sid[j]
                        and (gm is None or mod[j] == gm)]
            p = [j for j in eligible if pid[i] == pid[j]]
            n = [j for j in eligible if pid[i] != pid[j]]
            if p and n:
                pairs = [0.07 * torch.log1p(sum(torch.exp((s[i, k] - s[i, j] + 0.03) / 0.07)
                                               for k in n)) for j in p]
                queries.append(torch.stack(pairs).mean())
        result[name] = torch.stack(queries).mean() if queries else features.sum() * 0
    return result


def test_rank_formula_masks_and_gradients():
    torch.manual_seed(8)
    f = torch.randn(12, 24, requires_grad=True)
    pid, cam, mod, sid = metadata()
    sid[9] = sid[8]  # same raw image with a synthetic alternate camera: must still exclude
    actual, directions, stats = TAURRankingLoss()(f, pid, cam, mod, sid)
    expected = brute_rank(f, pid, cam, mod, sid)
    for key in directions:
        torch.testing.assert_close(directions[key], expected[key])
    torch.testing.assert_close(actual, sum(expected.values()) / 4)
    a = torch.autograd.grad(actual, f, retain_graph=True)[0]
    b = torch.autograd.grad(sum(expected.values()) / 4, f)[0]
    torch.testing.assert_close(a, b)
    assert stats['T2T']['queries'] == 2
    assert stats['R2T']['positive_pairs'] == 4


def test_reverse_gallery_and_empty_directions():
    f = torch.randn(12, 24, requires_grad=True)
    pid, cam, mod, sid = metadata()
    # R/N are fixed queries when gallery is TIR; gradients on unrelated R/N rows
    # are zero for the opposite reverse direction.
    _, directions, _ = TAURRankingLoss()(f, pid, cam, mod, sid)
    g = torch.autograd.grad(directions['R2T'], f)[0]
    assert g[4:8].count_nonzero() == 0
    assert g[8:].norm() > 0
    total, directions, stats = TAURRankingLoss()(f, pid, torch.zeros_like(cam), mod, sid)
    assert total.item() == 0 and all(v['queries'] == 0 for v in stats.values())
    total.backward()
    assert torch.isfinite(f.grad).all() and f.grad.count_nonzero() == 0


def test_missing_direction_keeps_coefficient_and_full_negative_counts():
    f = torch.randn(12, 24, requires_grad=True)
    pid, cam, mod, sid = metadata()
    cam[8:] = 2  # TIR has one camera: T2T empty, cross-modal directions remain valid
    total, directions, stats = TAURRankingLoss()(f, pid, cam, mod, sid)
    assert stats['T2T']['queries'] == 0 and stats['T2ALL']['queries'] == 4
    torch.testing.assert_close(total, sum(directions.values()) / 4)
    torch.testing.assert_close(directions['R2T'], brute_rank(f, pid, cam, mod, sid)['R2T'])
    assert stats['R2T']['negative_pairs'] == 8  # two negative images per query, no ID collapse


def test_manual_reduction_mean_unused_and_overflow(monkeypatch):
    import processor.taur_grad as module
    model = nn.Module()
    model.a = nn.Parameter(torch.ones(2))
    model.b = nn.Parameter(torch.ones(3))
    model.unused = nn.Parameter(torch.ones(1))
    model.a.grad = torch.tensor([2., 4.])
    # Peer has b gradients even when this rank has none.
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda: 2)
    calls = []

    def collective(tensor, op=None):
        calls.append(tensor.clone())
        if tensor.dtype == torch.int32:
            tensor.copy_(torch.tensor([1, 1, 0]))
        else:
            tensor.add_(torch.tensor([4., 6., 8., 10., float('inf')]))
    monkeypatch.setattr(module.dist, 'all_reduce', collective)
    synchronize_gradients(model)
    torch.testing.assert_close(model.a.grad, torch.tensor([3., 5.]))
    torch.testing.assert_close(model.b.grad[:2], torch.tensor([4., 5.]))
    assert torch.isinf(model.b.grad[2]) and model.unused.grad is None
    assert len(calls) == 2


def test_rank_fp32_under_autocast():
    f = torch.randn(12, 24, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        total, _, _ = TAURRankingLoss()(f, *metadata())
    assert total.dtype == torch.float32
    total.backward()
    assert torch.isfinite(f.grad).all()


def test_adapter_routing_cls_bound_and_per_image_norm():
    adapter = ThermalAdapter(24, 8)
    tokens = torch.randn(4, 5, 24, requires_grad=True)
    modes = torch.tensor([1, 2, 3, 3])
    output = adapter(tokens, modes)
    assert torch.equal(output[:2], tokens[:2])
    assert torch.equal(output[:, 0], tokens[:, 0])
    delta = (output[2:, 1:] - tokens[2:, 1:]).norm(dim=(1, 2))
    assert (delta <= 0.1 * tokens[2:, 1:].norm(dim=(1, 2)) + 1e-6).all()
    alone = adapter(tokens[2:3], modes[2:3])
    torch.testing.assert_close(alone, output[2:3])
    output.sum().backward()
    assert adapter.down.weight.grad.norm() > 0 and adapter.up.weight.grad.norm() > 0


def test_backbone_layers_and_disabled_equivalence():
    torch.manual_seed(4)
    original = tiny_backbone().eval()
    enabled = copy.deepcopy(original)
    enabled.enable_thermal_adapters([6, 9], 8, 0.1)
    x = torch.randn(3, 3, 32, 16)
    mod = torch.tensor([1, 2, 3])
    events = []
    handles = []
    for layer in (6, 9):
        handles.append(enabled.blocks[layer - 1].register_forward_hook(
            lambda m, inp, out, layer=layer: events.append('block' + str(layer))))
        handles.append(enabled.thermal_adapters[str(layer)].register_forward_hook(
            lambda m, inp, out, layer=layer: events.append('adapter' + str(layer))))
    a, b = original(x), enabled(x, modal_label=mod)
    assert torch.equal(a[:2], b[:2]) and not torch.equal(a[2], b[2])
    assert events == ['block6', 'adapter6', 'block9', 'adapter9']
    with pytest.raises(ValueError):
        enabled(x)
    for adapter in enabled.thermal_adapters.values():
        adapter.rho = 0
    assert torch.equal(a, enabled(x, modal_label=mod))
    for handle in handles:
        handle.remove()


def test_gradient_routing_through_later_shared_blocks():
    model = tiny_backbone()
    model.enable_thermal_adapters([6, 9], 8, 0.1)
    x = torch.randn(12, 3, 32, 16)
    pid, cam, mod, sid = metadata()
    f = model(x, modal_label=mod)
    base = f.square().mean()
    ranking, _, _ = TAURRankingLoss()(f, pid, cam, mod, sid)
    params = tuple(model.parameters())
    adapters = tuple(model.thermal_adapters.parameters())
    baseline = torch.autograd.grad(base, params, retain_graph=True, allow_unused=True)
    rank_grads = adapter_rank_gradients(ranking, adapters, torch.amp.GradScaler('cpu', enabled=False), 0.1)
    assert all(g is not None and g.norm() > 0 for g in rank_grads)
    base.backward()
    merge_adapter_gradients(adapters, rank_grads)
    rank_by_id = {id(p): g for p, g in zip(adapters, rank_grads)}
    for p, g in zip(params, baseline):
        expected = g if id(p) not in rank_by_id else g + rank_by_id[id(p)]
        if expected is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, expected)


def toy_data(counts=(8, 12, 8, 16, 4, 4, 8, 8)):
    return {m: [('{}_{}_{}.jpg'.format(m, p, i), p, i % (1 if p == 4 else 3), mid)
                for p, count in enumerate(counts) for i in range(count)]
            for mid, m in enumerate(['RGB', 'IR', 'Thermal'], 1)}


def test_camera_sampler_length_ddp_groups_seed_and_epoch():
    data = toy_data()
    modes = list(data)
    single = CameraPKMSampler(data, 16, 4, modes)
    rank0 = CameraPKMSampler(data, 16, 4, modes, rank=0, world_size=2)
    rank1 = CameraPKMSampler(data, 16, 4, modes, rank=1, world_size=2)
    a, b, c = list(single), list(rank0), list(rank1)
    assert len(a) == len(single) == 2 * len(b) == 2 * len(c)
    assert a == list(single)
    for step in range(single.steps):
        full = a[step * 16:(step + 1) * 16]
        assert full == b[step * 8:(step + 1) * 8] + c[step * 8:(step + 1) * 8]
        pids = []
        for start in range(0, 16, 4):
            group = full[start:start + 4]
            for mi, m in enumerate(modes):
                rows = [data[m][t[mi]] for t in group]
                assert len({r[1] for r in rows}) == 1
                assert len({r[2] for r in rows}) == (1 if rows[0][1] == 4 else 3)
                assert len({t[mi] for t in group}) == 4
            pids.append(data[modes[0]][group[0][0]][1])
        assert len(set(pids)) == 4
    single.set_epoch(2)
    assert list(single) != a


def test_metadata_collate_and_duplicate_paths(monkeypatch):
    import datasets.bases as bases
    monkeypatch.setattr(bases, 'read_image', lambda path: torch.ones(3, 32, 16))
    data = toy_data()
    ds = ImageDataset(data, list(data), return_metadata=True)
    batch = [ds[(0, 0, 0)], ds[(1, 1, 1)], ds[(0, 0, 0)]]
    imgs, pid, cam, mod, sid = taur_train_collate_fn(batch)
    assert mod.tolist() == [1] * 3 + [2] * 3 + [3] * 3
    assert sid[0] == sid[2] and sid[3] == sid[5] and sid[6] == sid[8]
    assert sid[0] != sid[3] and torch.cat(imgs).shape[0] == len(sid)
    assert pid.repeat(3).tolist() == [0] * 9


def test_schedule_optimizer_and_uad_auxiliary_contract():
    c = cfg.clone()
    c.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/TAUR.yml'))
    assert rank_weight(5, c.TAUR) == 0
    assert rank_weight(6, c.TAUR) == pytest.approx(0.1 / 15)
    assert rank_weight(20, c.TAUR) == rank_weight(120, c.TAUR) == 0.1
    m = tiny_backbone()
    m.enable_thermal_adapters([6, 9], 8)
    optimizer, _ = make_optimizer(c, m, nn.Linear(24, 2))
    groups = {id(g['params'][0]): g for g in optimizer.param_groups}
    assert all(groups[id(p)]['lr'] == 3 * c.SOLVER.BASE_LR
               for p in m.thermal_adapters.parameters())
    schedule = create_scheduler(c, optimizer)
    reference = next(i for i, group in enumerate(optimizer.param_groups)
                     if not group.get('taur_lr_factor') and schedule.base_values[i] == c.SOLVER.BASE_LR)
    for epoch in (0, 1, 4, 5, 20, 119, 120):
        schedule.step(epoch)
        assert all(groups[id(p)]['lr'] == pytest.approx(3 * optimizer.param_groups[reference]['lr'])
                   for p in m.thermal_adapters.parameters())
    f = torch.randn(24, 24, requires_grad=True)
    target = torch.tensor([0] * 4 + [1] * 4)
    pca, ids, centers = PCA(f, target, 3, 4)
    bank = PrototypeBank(2, 24, device='cpu')
    gpd = GPD(f, target, bank, ids, centers, 3)
    (pca + gpd).backward()
    bank.update(ids, centers, ddp=False)
    assert torch.isfinite(f.grad).all()
    torch.testing.assert_close(bank.prototypes.norm(dim=1), torch.ones(2))


def test_model_train_test_metadata_contract_and_checkpoint(tmp_path):
    c = cfg.clone()
    c.merge_from_file(str(Path(__file__).resolve().parents[1] / 'configs/TAUR.yml'))
    c.MODEL.PRETRAIN_CHOICE = 'none'
    c.INPUT.SIZE_TRAIN = [32, 16]
    factory = {'vit_base_in': lambda **kwargs: TransReID(
        img_size=(32, 16), embed_dim=768, num_heads=12, depth=12,
        stride_size=16, drop_path_rate=0)}
    model = build_transformer(2, 0, 0, c, factory).eval()
    imgs = [torch.randn(1, 3, 32, 16) for _ in range(3)]
    modes = torch.tensor([1, 2, 3])
    with torch.no_grad():
        _, train_f, _ = model(imgs, modality_ids=modes)
        test_f = torch.cat([model(img, mode=i + 1, modality_ids=modes[i:i + 1])
                            for i, img in enumerate(imgs)])
    torch.testing.assert_close(train_f, test_f, atol=2e-6, rtol=2e-5)
    state = {'module.' + k: v for k, v in model.state_dict().items()}
    torch.save(state, tmp_path / 'taur.pth')
    model.load_param(str(tmp_path / 'taur.pth'))
    torch.save({k: v for k, v in model.state_dict().items() if 'thermal_adapters.' not in k},
               tmp_path / 'uad.pth')
    with pytest.raises(ValueError, match='mismatch'):
        model.load_param(str(tmp_path / 'uad.pth'))
