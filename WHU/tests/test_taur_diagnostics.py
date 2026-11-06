"""Known-ranking, official-protocol and read-only inference regression checks."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from taur_diagnostics import (META, STATES, query_statistics, evaluate_protocol, aggregate,
                              assert_aligned, feature_effect, file_hash, export_features, state_hash)
from analyze_taur_ranking import run_analysis, paired_bootstrap
from diagnose_taur_checkpoint import EVALUATION, extraction_config, extract
from utils.metrics import eval_func, euclidean_distance
from model.backbones.vit_pytorch import TransReID


def pack():
    rng = np.random.RandomState(4)
    # Each identity is present on both platforms and in each modality.
    rows = [(split, pid, mid, cam) for split in ('query', 'gallery')
            for mid in (1, 2, 3) for pid in (1, 2) for cam in (0, 1, 5, 6)]
    n = len(rows)
    result = dict(features=rng.randn(n, 768).astype('float32'),
                  pid=np.array([r[1] for r in rows]), modid=np.array([r[2] for r in rows]),
                  camid=np.array([r[3] for r in rows]), split=np.array([r[0] for r in rows]),
                  path=np.array(['{}.jpg'.format(i) for i in range(n)]),
                  index=np.tile(np.arange(n // 2), 2), modality_index=np.tile(np.arange(8), 6))
    result['platform'] = np.where(result['camid'] < 5, 'G', 'A')
    return result


def test_adapter_runtime_bypass_has_no_state_or_weight_change():
    torch.manual_seed(3)
    base = TransReID(img_size=(32, 16), stride_size=16, embed_dim=24,
                    depth=12, num_heads=3, mlp_ratio=2, num_classes=0).eval()
    original = copy.deepcopy(base)
    base.enable_thermal_adapters([6, 9], 8, .1)
    before = state_hash(base)
    x, mids = torch.randn(3, 3, 32, 16), torch.tensor([1, 2, 3])
    calls = []
    handles = [m.register_forward_hook(lambda *args: calls.append(1)) for m in base.thermal_adapters.values()]
    with torch.inference_mode():
        on = base(x, modal_label=mids)
        assert len(calls) == 2
        calls.clear()
        base.adapter_enabled = False
        off = base(x, modal_label=mids)
        assert not calls
        assert torch.equal(off, original(x))
        assert torch.equal(on[:2], off[:2])
        assert not torch.equal(on[2], off[2])
        base.adapter_enabled = True
        assert torch.equal(on, base(x, modal_label=mids))
    assert state_hash(base) == before
    assert not any('adapter_enabled' in k for k in base.state_dict())
    for h in handles:
        h.remove()


def test_known_full_ranks_filter_all_same_camera_negatives_and_positives():
    p = dict(pid=np.array([1, 2, 1, 2, 1, 3, 1]), camid=np.array([0, 0, 0, 1, 1, 1, 1]),
             index=np.arange(7), path=np.array(list('abcdefg')), modid=np.ones(7, int),
             platform=np.array(['G'] * 7))
    scores = np.array([0, .99, .98, .9, .8, .7, .6])
    row = query_statistics(p, 0, np.arange(1, 7), scores)
    assert row['first_pos_rank'] == 2 and row['last_pos_rank'] == 4
    assert row['AP'] == .5 and row['INP'] == .5
    assert row['num_valid_pos'] == 2
    assert row['neg_before_pos_mean'] == 1.5 and row['neg_before_last_pos'] == 2
    assert row['wrong_id_before_pos_mean'] == 1.5
    assert row['C_img'] == 1.5 and row['C_id'] == 1.5
    assert row['M_best'] == pytest.approx(-.1)
    assert row['M_med'] == pytest.approx(-.2)


def test_ties_strict_score_counts_differ_from_rank_counts():
    p = dict(pid=np.array([1, 2, 1]), camid=np.array([0, 1, 1]), index=np.arange(3),
             path=np.array(['q', 'n', 'p']), modid=np.ones(3, int), platform=np.array(['G'] * 3))
    row = query_statistics(p, 0, np.array([1, 2]), np.array([0., .5, .5]))
    assert row['neg_before_pos_mean'] == 1
    assert row['C_img'] == row['C_id'] == 0
    assert row['score_ties'] == 1 and row['M_best'] == 0
    assert query_statistics(p, 0, np.array([1]), np.array([0., .5, .5])) is None
    positive_only = query_statistics(p, 0, np.array([2]), np.array([0., .5, .5]))
    assert positive_only['AP'] == positive_only['INP'] == 1
    assert np.isnan(positive_only['M_best'])


def test_full_gallery_positive_tail_beyond_top50():
    p = dict(pid=np.array([1] + [2] * 60 + [1]), camid=np.array([0] + [1] * 61),
             index=np.arange(62), path=np.array([str(i) for i in range(62)]),
             modid=np.ones(62, int), platform=np.array(['G'] * 62))
    row = query_statistics(p, 0, np.arange(1, 62), np.linspace(1, 0, 62))
    assert row['first_pos_rank'] == row['last_pos_rank'] == 61
    assert row['AP'] == row['INP'] == pytest.approx(1 / 61)
    assert row['wrong_id_before_pos_mean'] == row['C_id'] == 1
    assert row['C_img'] == 60


def test_independent_evaluator_matches_official_full_gallery():
    p = pack()
    q, g = p['split'] == 'query', p['split'] == 'gallery'
    rows = evaluate_protocol(p, q, g, chunk_size=5)
    f = torch.nn.functional.normalize(torch.from_numpy(p['features']), dim=1)
    order = torch.argsort(euclidean_distance(f[q], f[g]), dim=1).numpy()
    cmc, ap, inp = eval_func(order, p['pid'][q], p['pid'][g], p['camid'][q], p['camid'][g],
                            max_rank=10, return_minp=True)
    result = aggregate(rows)
    assert result['mAP'] == pytest.approx(ap * 100)
    assert result['mINP'] == pytest.approx(inp * 100)
    for k in (1, 5, 10):
        assert result['rank' + str(k)] == pytest.approx(cmc[k - 1] * 100, abs=1e-5)


def test_pairing_and_nonthermal_assertions():
    p = pack()
    packs = {s: copy.deepcopy(p) for s in STATES}
    packs['taur_off']['path'][0] = 'different'
    with pytest.raises(ValueError, match='metadata'):
        assert_aligned(packs)
    packs['taur_off'] = copy.deepcopy(p)
    packs['taur_off']['features'][0, 0] += .1
    with pytest.raises(AssertionError, match='RGB/NIR'):
        feature_effect(packs)


def save_fixture(directory):
    p = pack()
    manifest = dict(schema_version=1, evaluation=EVALUATION, states={})
    for state in STATES:
        path = directory / (state + '_features.npz')
        np.savez_compressed(path, **p)
        manifest['states'][state] = dict(checkpoint_sha256='same' if state != 'uad' else 'other',
                                        feature_sha256=file_hash(path))
    (directory / 'diagnostic_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_offline_end_to_end_all_protocols_and_outputs(tmp_path):
    save_fixture(tmp_path)
    result = run_analysis(tmp_path, chunk_size=3, bootstrap=10, plot=False)
    assert len(result) == 33  # 11 protocols x 3 states; 4 platform protocols separate
    for name in ('checkpoint_comparison', 'adapter_feature_effect', 'query_rank_transitions',
                 'ranking_summary', 'tir_ranking_mechanism', 'tir_platform_breakdown', 'tir_mixed_gallery_summary'):
        assert (tmp_path / (name + '.csv')).stat().st_size > 0
    assert [r for r in result if r['state'] == 'uad'] == [
        dict(r, state='uad') for r in result if r['state'] == 'taur_on']


def test_corrupted_cache_rejected(tmp_path):
    save_fixture(tmp_path)
    with (tmp_path / 'taur_on_features.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        run_analysis(tmp_path, bootstrap=0, plot=False)


def test_cluster_bootstrap_preserves_zero_delta():
    rows = [dict(pid=1, AP=.2, first_pos_rank=2), dict(pid=1, AP=.5, first_pos_rank=1),
            dict(pid=2, AP=.4, first_pos_rank=3)]
    result = paired_bootstrap(rows, rows, 20, 42)
    assert result['query_identities'] == 2
    assert result['delta_mAP_ci_low'] == result['delta_mAP_ci_high'] == 0
    assert result['delta_rank1_ci_low'] == result['delta_rank1_ci_high'] == 0


def test_extraction_config_cannot_enable_training_or_topk():
    root = Path(__file__).resolve().parents[1]
    config = extraction_config(str(root / 'configs/TAUR.yml'),
                              ['MODEL.DIST_TRAIN', 'True', 'TEST.TOP_K_EVAL', '50', 'TEST.NECK_FEAT', 'after'], True)
    assert config.MODEL.DIST_TRAIN is False and config.MODEL.PRETRAIN_CHOICE == 'none'
    assert config.TEST.TOP_K_EVAL == 0 and config.TEST.NECK_FEAT == 'before'


class FakeDataset(Dataset):
    def __init__(self, mid):
        self.modality = str(mid)
        self.dataset = {self.modality: [('/data/q{}.jpg'.format(mid), 1, 0, mid),
                                        ('/data/g{}.jpg'.format(mid), 1, 5, mid)]}

    def __len__(self):
        return 2

    def __getitem__(self, i):
        path, pid, cam, mid = self.dataset[self.modality][i]
        return torch.ones(768) * mid, pid, cam, cam, mid, Path(path).name


class FakeModel(nn.Module):
    def __init__(self, taur=False):
        super().__init__()
        self.base = nn.Module()
        self.base.thermal_adapters = nn.ModuleDict({'6': nn.Linear(1, 1), '9': nn.Linear(1, 1)}) if taur else nn.ModuleDict()
        self.base.adapter_enabled = True
        self.bn = nn.BatchNorm1d(768)

    def forward(self, images, mode, modality_ids):
        assert not self.training and not torch.is_grad_enabled()
        return images

    def load_param(self, path):
        pass


def test_export_metadata_query_gallery_order_and_no_updates():
    model = FakeModel()
    loaders = [DataLoader(FakeDataset(m), batch_size=1) for m in (1, 2, 3)]
    before = state_hash(model)
    p = export_features(model, loaders, [1, 1, 1], torch.device('cpu'))
    assert p['modid'].tolist() == [1, 2, 3, 1, 2, 3]
    assert p['split'].tolist() == ['query'] * 3 + ['gallery'] * 3
    assert p['platform'].tolist() == ['G'] * 3 + ['A'] * 3
    assert p['path'][0] == '/data/q1.jpg'
    assert state_hash(model) == before


def test_resume_cached_on_loads_taur_not_leftover_uad(tmp_path, monkeypatch):
    import diagnose_taur_checkpoint as module
    root = Path(__file__).resolve().parents[1]
    loaders = [DataLoader(FakeDataset(m), batch_size=1) for m in (1, 2, 3)]
    monkeypatch.setattr(module, 'make_dataloader', lambda cfg: (None, None, loaders, [1] * 3, 2, 7, 1))
    created = []
    def factory(cfg, **kwargs):
        created.append(cfg.TAUR.ENABLED)
        return FakeModel(cfg.TAUR.ENABLED)
    monkeypatch.setattr(module, 'make_model', factory)
    uad, taur = tmp_path / 'u.pth', tmp_path / 't.pth'
    uad.write_bytes(b'uad'); taur.write_bytes(b'taur')
    args = SimpleNamespace(uad_config=str(root / 'configs/UAD.yml'), taur_config=str(root / 'configs/TAUR.yml'),
                           opts=['MODEL.DEVICE', 'cpu'], output_dir=str(tmp_path), reuse=False,
                           uad_weight=str(uad), taur_weight=str(taur), extract_only=True)
    extract(args)
    assert created == [False, True]
    # Reextract UAD; reuse TAUR-on; off must explicitly load TAUR, not reuse UAD.
    (tmp_path / 'uad_features.npz').unlink()
    (tmp_path / 'taur_off_features.npz').unlink()
    args.reuse = True
    created.clear()
    extract(args)
    assert created == [False, True]
